#!/usr/bin/env bash
# Write env.local.sh and check that code, data, models, and interpreters are in place.
# This does not download the dataset or model weights.
set -euo pipefail
RELEASE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN_PYTHON=""
SIM_PYTHON=""
CHECK_ONLY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --train-python) TRAIN_PYTHON="$2"; shift 2 ;;
    --sim-python) SIM_PYTHON="$2"; shift 2 ;;
    --check) CHECK_ONLY=1; shift ;;
    -h|--help)
      echo "Usage: bash scripts/setup_env.sh [--train-python PATH] [--sim-python PATH] [--check]"
      exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

LOCAL="$RELEASE/env.local.sh"
if [[ ! -f "$LOCAL" ]]; then
  cat >"$LOCAL" <<EOF
# Local paths. This file stays on the machine and is not part of the code snapshot.
VLA_TRAIN_PYTHON=${TRAIN_PYTHON:-python3}
VLA_SIM_PYTHON=${SIM_PYTHON:-python3}
LINGBOT_REPO=\$RELEASE/third_party/lingbot-vla-v2
ROBOTWIN_ROOT=\$RELEASE/third_party/robotwin
VLA_DATA_ROOT=\$RELEASE/data/robotwin2_aloha_clean
VLA_MODEL_ROOT=\$RELEASE/models
VLA_BASE_MODEL=\$RELEASE/models/lingbot-vla-v2-6b
VLA_TOKENIZER=\$RELEASE/models/Qwen3-VL-4B-Instruct
VLA_RESULT_ROOT=\$RELEASE/results
VLA_GPU_IDS=
EOF
  # RELEASE is not expanded inside the heredoc on purpose when quoted; rewrite it.
  sed -i "s#\\\$RELEASE#$RELEASE#g" "$LOCAL"
  echo "Wrote $LOCAL"
else
  if [[ -n "$TRAIN_PYTHON" ]]; then
    if grep -q '^VLA_TRAIN_PYTHON=' "$LOCAL"; then
      sed -i "s#^VLA_TRAIN_PYTHON=.*#VLA_TRAIN_PYTHON=$TRAIN_PYTHON#" "$LOCAL"
    else
      echo "VLA_TRAIN_PYTHON=$TRAIN_PYTHON" >>"$LOCAL"
    fi
  fi
  if [[ -n "$SIM_PYTHON" ]]; then
    if grep -q '^VLA_SIM_PYTHON=' "$LOCAL"; then
      sed -i "s#^VLA_SIM_PYTHON=.*#VLA_SIM_PYTHON=$SIM_PYTHON#" "$LOCAL"
    else
      echo "VLA_SIM_PYTHON=$SIM_PYTHON" >>"$LOCAL"
    fi
  fi
fi

# shellcheck disable=SC1091
source "$RELEASE/scripts/env.sh"
mkdir -p "$VLA_DATA_ROOT" "$VLA_MODEL_ROOT" "$VLA_RESULT_ROOT" "$RELEASE/third_party"

fail=0
note() { echo "MISSING  $1"; fail=1; }
ok() { echo "OK       $1"; }

[[ -x "$VLA_TRAIN_PYTHON" || "$VLA_TRAIN_PYTHON" == "python3" ]] || note "VLA_TRAIN_PYTHON is not executable: $VLA_TRAIN_PYTHON"
[[ -d "$LINGBOT_REPO/lingbotvla" ]] || note "model source: $LINGBOT_REPO (expected a lingbot-vla-v2 checkout)"
[[ -d "$ROBOTWIN_ROOT/envs" ]] || note "simulator source: $ROBOTWIN_ROOT"
[[ -f "$VLA_DATA_ROOT/data_index.json" ]] || note "dataset index: $VLA_DATA_ROOT/data_index.json"
[[ -d "$VLA_BASE_MODEL" ]] || note "base checkpoint: $VLA_BASE_MODEL"
[[ -d "$VLA_TOKENIZER" ]] || note "tokenizer: $VLA_TOKENIZER"
[[ -f "$VLA_MOGE" ]] || note "depth teacher: $VLA_MOGE"
[[ -f "$VLA_MORGBD" ]] || note "depth teacher: $VLA_MORGBD"
[[ -f "$VLA_DINO_CKPT" ]] || note "video teacher: $VLA_DINO_CKPT"
[[ -f /usr/share/nvidia/nvoptix.bin ]] || note "OptiX denoiser weights /usr/share/nvidia/nvoptix.bin"
[[ -e /usr/lib/x86_64-linux-gnu/libGLX_nvidia.so.0 ]] || note "NVIDIA GLX library libGLX_nvidia.so.0"
[[ -e /usr/lib/x86_64-linux-gnu/libnvoptix.so.1 ]] || note "NVIDIA OptiX library libnvoptix.so.1"

if [[ "$fail" -eq 0 ]]; then
  ok "layout"
else
  echo "Fill the missing items, then rerun: bash scripts/setup_env.sh --check"
fi

if [[ "$CHECK_ONLY" -eq 1 || "$fail" -eq 0 ]]; then
  "$VLA_TRAIN_PYTHON" - <<'PY' || fail=1
import importlib, sys
mods = ["torch", "transformers", "yaml", "safetensors"]
bad = []
for name in mods:
    try:
        importlib.import_module(name)
    except Exception as exc:
        bad.append(f"{name}: {exc.__class__.__name__}")
print("train-python", sys.executable)
if bad:
    print("MISSING python modules:", ", ".join(bad))
    raise SystemExit(1)
print("OK train imports")
PY
fi
if [[ "$fail" -ne 0 ]]; then
  exit 1
fi
echo "Environment check passed."
