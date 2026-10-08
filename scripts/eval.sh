#!/usr/bin/env bash
# Clean evaluation. Example:
#   bash scripts/eval.sh --checkpoint results/<run>/checkpoints/global_step_5000/hf_ckpt
set -euo pipefail
RELEASE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$RELEASE/scripts/env.sh"
if [[ -z "${VLA_GPU_IDS}" ]]; then
  echo "Set VLA_GPU_IDS in env.local.sh before evaluation." >&2
  exit 2
fi
exec "$VLA_TRAIN_PYTHON" "$RELEASE/eval/launch_eval.py" "$@"
