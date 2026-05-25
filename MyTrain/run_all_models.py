import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def append_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(text)


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
    parser = argparse.ArgumentParser(description="Sequential trainer for Deeplabv3+, DDUnet, AttentionUNet, VNet.")
    parser.add_argument("--root", type=str, default=None, help="Project root directory")
    parser.add_argument("--state", type=str, default="runner_state.json", help="State file under MyTrain")
    parser.add_argument("--log-dir", type=str, default="logs", help="Log directory under MyTrain")
    parser.add_argument("--rerun-completed", action="store_true", help="Run models even if state marks them completed")
    args = parser.parse_args()

    if args.root:
        if platform.system() != "Windows" and ":" in args.root:
            raise ValueError(
                f"Invalid --root for non-Windows environment: {args.root}. "
                "Please use an absolute Linux path like /home/intern/ftp/wusi/Project_crop/Monai_model."
            )
        root = Path(args.root).expanduser().resolve()
    else:
        # Default to the parent of this script's directory: <root>/MyTrain/run_all_models.py
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

    state = load_state(state_path)
    state.setdefault("models", {})

    append_text(master_log, f"\n[{now_str()}] Runner started | root={root}\n")

    env = dict(os.environ)

    for m in models:
        name = m["name"]
        run_py = m["run_py"]
        workdir = m["workdir"]
        model_log = log_dir / f"{name.replace('+', 'plus').lower()}.log"

        if not run_py.exists():
            msg = f"[{now_str()}] SKIP {name} | run.py not found: {run_py}\n"
            append_text(master_log, msg)
            state["models"][name] = {
                "status": "missing",
                "last_run": now_str(),
                "return_code": None,
                "run_py": str(run_py),
            }
            save_state(state_path, state)
            continue

        if (not args.rerun_completed) and state["models"].get(name, {}).get("status") == "completed":
            append_text(master_log, f"[{now_str()}] SKIP {name} | already completed in state\n")
            continue

        try:
            rc = run_model(name, run_py, workdir, model_log, env)
            status = "completed" if rc == 0 else "failed"
            state["models"][name] = {
                "status": status,
                "last_run": now_str(),
                "return_code": rc,
                "run_py": str(run_py),
                "log_file": str(model_log),
            }
            save_state(state_path, state)
            append_text(master_log, f"[{now_str()}] {name} finished | status={status} | rc={rc} | log={model_log}\n")
        except KeyboardInterrupt:
            state["models"].setdefault(name, {})
            state["models"][name].update({
                "status": "interrupted",
                "last_run": now_str(),
                "run_py": str(run_py),
                "log_file": str(model_log),
            })
            save_state(state_path, state)
            append_text(master_log, f"[{now_str()}] INTERRUPTED at {name}. Re-run script to resume.\n")
            raise
        except Exception as e:
            state["models"][name] = {
                "status": "runner_error",
                "last_run": now_str(),
                "error": repr(e),
                "run_py": str(run_py),
                "log_file": str(model_log),
            }
            save_state(state_path, state)
            append_text(master_log, f"[{now_str()}] Runner exception on {name}: {repr(e)}\n")
            # continue to next model as requested
            continue

    append_text(master_log, f"[{now_str()}] Runner finished\n")

    failed = [k for k, v in state["models"].items() if v.get("status") not in ("completed",)]
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
