#!/usr/bin/env bash
# Train from the raw LingBot-VLA checkpoint for VLA_STEPS optimizer steps (default 5000).
set -euo pipefail
RELEASE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$RELEASE/scripts/env.sh"
if [[ -z "${VLA_GPU_IDS}" ]]; then
  echo "Set VLA_GPU_IDS in env.local.sh. The reported 5000-step run used 3 GPUs, micro-batch 8, global batch 120." >&2
  exit 2
fi
exec "$VLA_TRAIN_PYTHON" "$RELEASE/train/launch.py" "$@"
