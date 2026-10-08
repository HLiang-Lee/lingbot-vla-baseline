#!/usr/bin/env python3
"""Launch clean-only training from the raw LingBot checkpoint. Refuses busy GPUs."""
import datetime, json, os, pathlib, shutil, subprocess, sys

import yaml

RELEASE = pathlib.Path(__file__).resolve().parents[1]
TRAIN = RELEASE / "train"


def need(name):
    value = os.environ.get(name)
    if not value:
        sys.exit(f"Missing environment variable {name}")
    return value


def gpu_inventory():
    proc = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,memory.used", "--format=csv,noheader"],
        capture_output=True, text=True, check=True)
    rows = {}
    for line in proc.stdout.splitlines():
        index, uuid, used = [part.strip() for part in line.split(",")]
        rows[int(index)] = {"uuid": uuid, "memory_mib": int(used.split()[0])}
    return rows


def busy_uuids():
    proc = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"],
        capture_output=True, text=True, check=True)
    return {line.strip() for line in proc.stdout.splitlines() if line.strip()}


def main():
    gpus = [int(part) for part in need("VLA_GPU_IDS").split(",") if part.strip() != ""]
    if not gpus or len(gpus) != len(set(gpus)):
        sys.exit("VLA_GPU_IDS must be a comma-separated list of distinct GPU indices")
    inventory = gpu_inventory()
    missing = [gpu for gpu in gpus if gpu not in inventory]
    if missing:
        sys.exit(f"GPU indices not present: {missing}")
    occupied = [gpu for gpu in gpus if inventory[gpu]["uuid"] in busy_uuids()]
    if occupied:
        sys.exit(f"Refusing busy GPUs: {occupied}")

    micro = int(os.environ.get("VLA_MICRO_BATCH", "8"))
    global_batch = int(os.environ.get("VLA_GLOBAL_BATCH", "120"))
    steps = int(os.environ.get("VLA_STEPS", "5000"))
    save_steps = int(os.environ.get("VLA_SAVE_STEPS", "1000"))
    if global_batch % (len(gpus) * micro) != 0:
        sys.exit("global batch must be divisible by GPU count times micro-batch")
    accum = global_batch // (len(gpus) * micro)

    data_root = pathlib.Path(need("VLA_DATA_ROOT"))
    index = data_root / "data_index.json"
    if not index.exists():
        sys.exit(f"Missing data index: {index}")
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run = pathlib.Path(need("VLA_RESULT_ROOT")) / f"baseline_{stamp}"
    run.mkdir(parents=True, exist_ok=False)
    shutil.copytree(TRAIN / "src", run / "code")
    robot_dir = run / "robot_configs"
    robot_dir.mkdir()
    robot = (TRAIN / "robot_configs" / "robotwin.yaml").read_text().replace(
        "__NORM_STATS__", str(run / "clean_norm_stats.json"))
    (robot_dir / "robotwin.yaml").write_text(robot)
    shutil.copy2(TRAIN / "norm" / "clean_norm_stats.json", run / "clean_norm_stats.json")
    shutil.copy2(index, run / "data_index.json")

    text = (TRAIN / "configs" / "baseline_5000.yaml").read_text()
    replacements = {
        "__BASE_MODEL__": need("VLA_BASE_MODEL"),
        "__TOKENIZER__": need("VLA_TOKENIZER"),
        "__DATA_INDEX__": str(run / "data_index.json"),
        "__ROBOT_CONFIG_ROOT__": str(robot_dir),
        "__NORM_STATS__": str(run / "clean_norm_stats.json"),
        "__OUTPUT_DIR__": str(run),
        "__MOGE__": need("VLA_MOGE"),
        "__MORGBD__": need("VLA_MORGBD"),
        "__DINO_CKPT__": need("VLA_DINO_CKPT"),
        "__DINO_CFG__": need("VLA_DINO_CFG"),
    }
    for token, value in replacements.items():
        if token not in text:
            sys.exit(f"Config token disappeared: {token}")
        text = text.replace(token, value)
    cfg = yaml.safe_load(text)
    cfg["train"]["max_steps"] = steps
    cfg["train"]["save_steps"] = save_steps
    cfg["train"]["micro_batch_size"] = micro
    cfg["train"]["global_batch_size"] = global_batch
    cfg["train"]["gradient_accumulation_steps"] = accum
    (run / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    python = need("VLA_TRAIN_PYTHON")
    repo = need("LINGBOT_REPO")
    command = [
        python, "-u", "-m", "torch.distributed.run", "--standalone",
        f"--nproc_per_node={len(gpus)}", "--log-dir", str(run / "ranks"),
        "--redirects", "3", "--tee", "3", str(run / "code" / "train.py"), str(run / "config.yaml"),
    ]
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": ",".join(str(gpu) for gpu in gpus),
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "PYTHONPATH": str(run / "code") + os.pathsep + repo,
        "PYTHONNOUSERSITE": "1",
        "PYTHONUNBUFFERED": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "ROUND1_RUN_DIR": str(run),
        "ROUND1_SAMPLING": "frame",
        "ROUND1_AUGMENT": "0",
        "ROUND1_BACKBONE_LR_SCALE": "1",
        "ROUND1_TRAIN_EXPERT_ONLY": "0",
        "QWEN3VL_PATH": need("VLA_TOKENIZER"),
        "QWEN3_PATH": need("VLA_TOKENIZER"),
    })
    (run / "launch.json").write_text(json.dumps({
        "gpus": gpus,
        "steps": steps,
        "micro_batch": micro,
        "global_batch": global_batch,
        "accumulation": accum,
        "base_model": need("VLA_BASE_MODEL"),
        "data_root": str(data_root),
    }, indent=2))
    log = open(run / "train.log", "a")
    proc = subprocess.Popen(command, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, start_new_session=True)
    (run / "state.json").write_text(json.dumps({"status": "running", "pid": proc.pid}))
    print(json.dumps({"pid": proc.pid, "run_dir": str(run), "steps": steps}, indent=2))


if __name__ == "__main__":
    main()
