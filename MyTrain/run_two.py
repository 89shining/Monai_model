import argparse
import concurrent.futures
import json
import os
import platform
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def append_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(text)


def log_both(path: Path, text: str) -> None:
    print(text, end="", flush=True)
    append_text(path, text)


def load_state(state_path: Path) -> Dict:
    if not state_path.exists():
        return {"models": {}}
    with state_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_state(state_path: Path, state: Dict) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with state_path.open("w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def run_model(name: str, run_py: Path, workdir: Path, log_file: Path, env: Dict[str, str]) -> int:
    header = (
        "\n" + "=" * 100 + "\n"
        f"[{now_str()}] START {name}\n"
        f"workdir={workdir}\n"
        f"cmd={sys.executable} {run_py}\n"
        f"CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES', '')}\n"
        + "=" * 100 + "\n"
    )
    append_text(log_file, header)

    with log_file.open("a", encoding="utf-8") as f:
        proc = subprocess.run(
            [sys.executable, str(run_py)],
            cwd=str(workdir),
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            check=False,
        )

    footer = (
        f"\n[{now_str()}] END {name} | return_code={proc.returncode}\n"
        + "-" * 100 + "\n"
    )
    append_text(log_file, footer)
    return proc.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Two-GPU grouped runner: group-parallel, in-group sequential.")
    parser.add_argument("--root", type=str, default=None, help="Project root directory")
    parser.add_argument("--state", type=str, default="runner_state.json", help="State file under MyTrain")
    parser.add_argument("--log-dir", type=str, default="logs", help="Log directory under MyTrain")
    parser.add_argument("--rerun-completed", action="store_true", help="Run models even if state marks them completed")
    parser.add_argument("--gpu-a", type=str, default="0", help="GPU for group A: VNet -> Deeplabv3+")
    parser.add_argument("--gpu-b", type=str, default="1", help="GPU for group B: AttentionUNet -> DDUnet")
    args = parser.parse_args()

    if args.root:
        if platform.system() != "Windows" and ":" in args.root:
            raise ValueError(
                f"Invalid --root for non-Windows environment: {args.root}. "
                "Please use an absolute Linux path like /home/intern/ftp/wusi/Project_crop/Monai_model."
            )
        root = Path(args.root).expanduser().resolve()
    else:
        root = Path(__file__).resolve().parent.parent

    mytrain = root / "MyTrain"
    state_path = mytrain / args.state
    log_dir = mytrain / args.log_dir
    master_log = log_dir / "master.log"

    models: List[Dict[str, Path]] = [
        {"name": "Deeplabv3+", "run_py": root / "Deeplabv3+" / "run.py", "workdir": root / "Deeplabv3+"},
        {"name": "DDUnet", "run_py": root / "DDUnet" / "run.py", "workdir": root / "DDUnet"},
        {"name": "AttentionUNet", "run_py": root / "AttentionUNet" / "run.py", "workdir": root / "AttentionUNet"},
        {"name": "VNet", "run_py": root / "VNet" / "run.py", "workdir": root / "VNet"},
    ]
    models_by_name = {m["name"]: m for m in models}

    groups = {
        "A": {"gpu": args.gpu_a, "models": ["VNet", "Deeplabv3+"]},
        "B": {"gpu": args.gpu_b, "models": ["AttentionUNet", "DDUnet"]},
    }

    state = load_state(state_path)
    state.setdefault("models", {})

    log_both(master_log, f"\n[{now_str()}] Runner started | root={root}\n")

    base_env = dict(os.environ)
    io_lock = threading.Lock()

    def run_group(group_name: str, gpu_id: str, ordered_models: List[str]) -> None:
        for model_name in ordered_models:
            m = models_by_name[model_name]
            name = m["name"]
            run_py = m["run_py"]
            workdir = m["workdir"]
            model_log = log_dir / f"{name.replace('+', 'plus').lower()}.log"

            if not run_py.exists():
                with io_lock:
                    log_both(master_log, f"[{now_str()}] SKIP {name} | run.py not found: {run_py}\n")
                    state["models"][name] = {
                        "status": "missing",
                        "last_run": now_str(),
                        "return_code": None,
                        "run_py": str(run_py),
                    }
                    save_state(state_path, state)
                continue

            if (not args.rerun_completed) and state["models"].get(name, {}).get("status") == "completed":
                with io_lock:
                    log_both(master_log, f"[{now_str()}] SKIP {name} | already completed in state\n")
                continue

            env = dict(base_env)
            env["CUDA_VISIBLE_DEVICES"] = gpu_id

            with io_lock:
                log_both(master_log, f"[{now_str()}] RUN {name} ... | group={group_name} | gpu={gpu_id}\n")

            rc = run_model(name, run_py, workdir, model_log, env)
            status = "completed" if rc == 0 else "failed"

            with io_lock:
                state["models"][name] = {
                    "status": status,
                    "last_run": now_str(),
                    "return_code": rc,
                    "run_py": str(run_py),
                    "log_file": str(model_log),
                }
                save_state(state_path, state)
                log_both(master_log, f"[{now_str()}] {name} finished | status={status} | rc={rc} | log={model_log}\n")

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            fa = executor.submit(run_group, "A", groups["A"]["gpu"], groups["A"]["models"])
            fb = executor.submit(run_group, "B", groups["B"]["gpu"], groups["B"]["models"])
            fa.result()
            fb.result()
    except KeyboardInterrupt:
        with io_lock:
            log_both(master_log, f"[{now_str()}] INTERRUPTED. Re-run script to resume.\n")
        raise
    except Exception as e:
        with io_lock:
            log_both(master_log, f"[{now_str()}] Runner exception: {repr(e)}\n")
        raise

    log_both(master_log, f"[{now_str()}] Runner finished\n")

    failed = [k for k, v in state["models"].items() if v.get("status") not in ("completed",)]
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
