import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict


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
        f"CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES', '')}\n"
        + "=" * 100 + "\n"
    )
    append_text(log_file, header)

    with log_file.open("a", encoding="utf-8") as f:
        proc = subprocess.Popen(
            [sys.executable, str(run_py)],
            cwd=str(workdir),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            f.write(line)
        proc.wait()

    footer = (
        f"\n[{now_str()}] END {name} | return_code={proc.returncode}\n"
        + "-" * 100 + "\n"
    )
    append_text(log_file, footer)
    return int(proc.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(description="Sequential runner: VNet -> AttentionUNet")
    parser.add_argument("--root", type=str, default=None, help="Project root directory")
    parser.add_argument("--log-dir", type=str, default="logs", help="Log directory under MyTrain")
    parser.add_argument("--state", type=str, default="runner_vnet_attention_state.json", help="State file under MyTrain")
    parser.add_argument("--gpu", type=str, default=os.environ.get("CUDA_VISIBLE_DEVICES", "1"), help="GPU id")
    parser.add_argument("--rerun-completed", action="store_true", help="Run model even if completed in state")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve() if args.root else Path(__file__).resolve().parent.parent
    mytrain = root / "MyTrain"
    log_dir = mytrain / args.log_dir
    state_path = mytrain / args.state
    master_log = log_dir / "vnet_attention_master.log"

    models = [
        {"name": "VNet", "run_py": root / "VNet" / "run.py", "workdir": root / "VNet", "log": log_dir / "vnet.log"},
        {"name": "AttentionUNet", "run_py": root / "AttentionUNet" / "run.py", "workdir": root / "AttentionUNet", "log": log_dir / "attentionunet.log"},
    ]

    state = load_state(state_path)
    state.setdefault("models", {})

    append_text(master_log, f"\n[{now_str()}] Runner started | root={root} | gpu={args.gpu}\n")

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = args.gpu

    for m in models:
        name = m["name"]
        run_py = m["run_py"]
        workdir = m["workdir"]
        log_file = m["log"]

        if not run_py.exists():
            append_text(master_log, f"[{now_str()}] SKIP {name} | run.py not found: {run_py}\n")
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

        append_text(master_log, f"[{now_str()}] RUN {name}\n")
        rc = run_model(name, run_py, workdir, log_file, env)
        status = "completed" if rc == 0 else "failed"

        state["models"][name] = {
            "status": status,
            "last_run": now_str(),
            "return_code": rc,
            "run_py": str(run_py),
            "log_file": str(log_file),
        }
        save_state(state_path, state)

        append_text(master_log, f"[{now_str()}] {name} finished | status={status} | rc={rc} | log={log_file}\n")

        if rc != 0:
            append_text(master_log, f"[{now_str()}] STOP chain because {name} failed\n")
            return rc

    append_text(master_log, f"[{now_str()}] Runner finished\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
