#!/usr/bin/env bash
set -euo pipefail

# Usage example:
# bash scripts/eval_gaet.sh task1_task2_task3 EEG EEG 0
# bash scripts/eval_gaet.sh task1_task2_taskNRv2 noise noise 1
#
# Args:
#   1) TASK_NAME: task1_task2_task3 | task1_task2_taskNRv2
#   2) TRAIN_INPUT: EEG | noise
#   3) TEST_INPUT: EEG | noise
#   4) GPU_ID: visible gpu index

TASK_NAME="${1:-task1_task2_task3}"
TRAIN_INPUT="${2:-EEG}"
TEST_INPUT="${3:-EEG}"
GPU_ID="${4:-0}"

SAVE_NAME="${TASK_NAME}_gaet_b16_s120_s230_lr0.0001_2e-05_${TRAIN_INPUT}"
CHECKPOINT_PATH="checkpoints/decoding/best/${SAVE_NAME}.pt"
CONFIG_PATH="config/decoding/${SAVE_NAME}.json"

CUDA_VISIBLE_DEVICES="${GPU_ID}" python3 eval_decoding.py \
  --checkpoint_path "${CHECKPOINT_PATH}" \
  --config_path "${CONFIG_PATH}" \
  --data_root ./dataset/ZuCo \
  --test_input "${TEST_INPUT}" \
  --train_input "${TRAIN_INPUT}" \
  -cuda cuda:0
