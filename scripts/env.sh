#!/usr/bin/env bash
# Shared path defaults for this release. Override in env.local.sh.
set -euo pipefail
RELEASE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "$RELEASE/env.local.sh" ]]; then
  # shellcheck disable=SC1091
  source "$RELEASE/env.local.sh"
fi

: "${LINGBOT_REPO:=$RELEASE/third_party/lingbot-vla-v2}"
: "${ROBOTWIN_ROOT:=$RELEASE/third_party/robotwin}"
: "${VLA_DATA_ROOT:=$RELEASE/data/robotwin2_aloha_clean}"
: "${VLA_MODEL_ROOT:=$RELEASE/models}"
: "${VLA_BASE_MODEL:=$VLA_MODEL_ROOT/lingbot-vla-v2-6b}"
: "${VLA_TOKENIZER:=$VLA_MODEL_ROOT/Qwen3-VL-4B-Instruct}"
: "${VLA_RESULT_ROOT:=$RELEASE/results}"
: "${VLA_MOGE:=$VLA_MODEL_ROOT/moge-2-vitb-normal/model.pt}"
: "${VLA_MORGBD:=$VLA_BASE_MODEL/depth/model.pt}"
: "${VLA_DINO_CKPT:=$VLA_BASE_MODEL/dino_video/teacher_step_10000.pth}"
: "${VLA_DINO_CFG:=$VLA_BASE_MODEL/dino_video/config.yaml}"
: "${VLA_TRAIN_PYTHON:=python3}"
: "${VLA_SIM_PYTHON:=python3}"
: "${VLA_STEPS:=5000}"
: "${VLA_SAVE_STEPS:=1000}"
: "${VLA_MICRO_BATCH:=8}"
: "${VLA_GLOBAL_BATCH:=120}"
: "${VLA_GPU_IDS:=}"

export RELEASE LINGBOT_REPO ROBOTWIN_ROOT VLA_DATA_ROOT VLA_MODEL_ROOT VLA_BASE_MODEL
export VLA_TOKENIZER VLA_RESULT_ROOT VLA_MOGE VLA_MORGBD VLA_DINO_CKPT VLA_DINO_CFG
export VLA_TRAIN_PYTHON VLA_SIM_PYTHON VLA_STEPS VLA_SAVE_STEPS VLA_MICRO_BATCH VLA_GLOBAL_BATCH VLA_GPU_IDS
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
