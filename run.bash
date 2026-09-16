#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_root"

dataset_path="${DATASET_PATH:-datasets/librispeech_sampled_600_file_0s_4s}"
model_name="${MODEL_NAME:-ds1}"
max_iterations="${MAX_ITERATIONS:-2000}"
learning_rate="${LEARNING_RATE:-0.5}"

python src/main.py \
  --model_name "$model_name" \
  --dataset_path "$dataset_path" \
  --batch_start_idx "${BATCH_START_IDX:-0}" \
  --batch_end_idx "${BATCH_END_IDX:-1}" \
  --min_duration_ms "${MIN_DURATION_MS:-0}" \
  --max_duration_ms "${MAX_DURATION_MS:-13000}" \
  --learning_rate "$learning_rate" \
  --max_iterations "$max_iterations" \
  --context_frames "${CONTEXT_FRAMES:-6}" \
  --dropout_prob "${DROPOUT_PROB:-0.0}" \
  --use_grid_optimization \
  --grid_size "${GRID_SIZE:-300}" \
  --grid_overlap "${GRID_OVERLAP:-150}"
