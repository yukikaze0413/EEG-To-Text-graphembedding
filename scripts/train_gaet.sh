#!/usr/bin/env bash
set -euo pipefail

# Usage example:
# bash scripts/train_gaet.sh task1_task2_task3 EEG 0
# bash scripts/train_gaet.sh task1_task2_taskNRv2 noise 1

TASK_NAME="${1:-task1_task2_task3}"   # task1_task2_task3 | task1_task2_taskNRv2
TRAIN_INPUT="${2:-EEG}"               # EEG | noise
GPU_ID="${3:-0}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" python3 train_decoding.py \
  --task_name "${TASK_NAME}" \
  --train_input "${TRAIN_INPUT}" \
  --data_root ./dataset/ZuCo \
  -b 16 \
  --stage1_epochs 20 \
  --stage2_epochs 30 \
  --stage1_lr 1e-4 \
  --stage2_lr 2e-5 \
  --llm_name facebook/bart-large \
  --num_query_tokens 32 \
  -cuda cuda:0 \
  -s ./checkpoints/decoding
