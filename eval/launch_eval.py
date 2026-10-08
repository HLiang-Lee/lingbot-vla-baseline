#!/usr/bin/env python3
"""Launch a clean evaluation. Refuses busy GPUs and does not resume a previous run."""
import datetime, json, os, pathlib, subprocess, sys

RELEASE = pathlib.Path(__file__).resolve().parents[1]


def need(name):
    value = os.environ.get(name)
    if not value:
        sys.exit(f"Missing environment variable {name}")
    return value


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--execute", type=int, choices=[10, 25, 50], default=50)
    parser.add_argument("--num-tasks", type=int, default=50)
    parser.add_argument("--gpu-ids", default=os.environ.get("VLA_GPU_IDS", ""))
    parser.add_argument("--port", type=int, default=29110)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()
    if not (checkpoint / "model.safetensors.index.json").exists():
        sys.exit("Pass an exported hf_ckpt directory")
    gpu_ids = [int(part) for part in args.gpu_ids.split(",") if part.strip() != ""]
    if not gpu_ids or len(gpu_ids) != len(set(gpu_ids)):
        sys.exit("Set VLA_GPU_IDS or pass --gpu-ids")
    if not 1 <= len(gpu_ids) <= 5:
        sys.exit("This evaluator accepts 1 to 5 GPUs")
    apps = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"],
        capture_output=True, text=True, check=True).stdout
    listed = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
        capture_output=True, text=True, check=True).stdout
    uuids = {}
    for line in listed.splitlines():
        index, uuid = [part.strip() for part in line.split(",")]
        uuids[int(index)] = uuid
    occupied = [gpu for gpu in gpu_ids if uuids.get(gpu, "") and uuids[gpu] in apps]
    if occupied:
        sys.exit(f"Refusing busy GPUs: {occupied}")
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run = pathlib.Path(need("VLA_RESULT_ROOT")) / f"{stamp}_execute{args.execute}"
    command = [
        need("VLA_TRAIN_PYTHON"), "-u", str(RELEASE / "eval" / "evaluate.py"),
        "--run-dir", str(run), "--episodes", str(args.episodes),
        "--num-tasks", str(args.num_tasks), "--gpus", str(len(gpu_ids)),
        "--gpu-ids", ",".join(map(str, gpu_ids)), "--port", str(args.port),
    ]
    env = os.environ.copy()
    env.update({
        "ROUND1_EVAL_MODEL": str(checkpoint),
        "ROUND1_EXECUTE_LENGTH": str(args.execute),
        "ROUND1_EVAL_SETTINGS": "demo_clean",
        "LINGBOT_REPO": need("LINGBOT_REPO"),
        "ROBOTWIN_ROOT": need("ROBOTWIN_ROOT"),
        "VLA_TRAIN_PYTHON": need("VLA_TRAIN_PYTHON"),
        "VLA_SIM_PYTHON": need("VLA_SIM_PYTHON"),
        "VLA_TOKENIZER": need("VLA_TOKENIZER"),
        "VLA_MODEL_ROOT": need("VLA_MODEL_ROOT"),
    })
    if args.dry_run:
        print(json.dumps({"command": command, "run_dir": str(run), "settings": "demo_clean"}, indent=2))
        return
    run.mkdir(parents=True, exist_ok=False)
    (run / "round1_eval_launch.json").write_text(json.dumps({
        "checkpoint": str(checkpoint), "execute": args.execute, "episodes": args.episodes,
        "settings": "demo_clean", "gpu_ids": gpu_ids,
    }, indent=2))
    log = open(run / "scheduler.log", "a")
    proc = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, start_new_session=True)
    print(json.dumps({"pid": proc.pid, "run_dir": str(run)}, indent=2))


if __name__ == "__main__":
    main()
