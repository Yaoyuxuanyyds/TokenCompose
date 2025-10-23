#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

##############################
# User configuration section #
##############################
# Path to the JSONL/JSON captions file used as preprocessing input. Each
# entry should contain `img_path` and `caption` fields as described in
# preprocess_data/readme.md.
INPUT_JSON=${INPUT_JSON:-/absolute/path/to/input.jsonl}

# Root directory for the HuggingFace ImageFolder dataset. The script will
# write processed assets into ${DATASET_ROOT}/${SPLIT_NAME}.
DATASET_ROOT=${DATASET_ROOT:-/absolute/path/to/tokencompose_dataset}
SPLIT_NAME=${SPLIT_NAME:-train}

# Training output directory.
OUTPUT_DIR=${OUTPUT_DIR:-${REPO_ROOT}/results/boundary_supervision}

# Stable Diffusion base model and hyper-parameters.
PRETRAINED_MODEL=${PRETRAINED_MODEL:-CompVis/stable-diffusion-v1-4}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-1}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-4}
MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS:-20000}
LEARNING_RATE=${LEARNING_RATE:-5e-6}
TOKEN_LOSS_SCALE=${TOKEN_LOSS_SCALE:-1e-3}
PIXEL_LOSS_SCALE=${PIXEL_LOSS_SCALE:-5e-5}
GEO_LOSS_ALPHA=${GEO_LOSS_ALPHA:-0.2}
GEO_BOUNDARY_WEIGHT=${GEO_BOUNDARY_WEIGHT:-1.0}
GEO_SMOOTH_WEIGHT=${GEO_SMOOTH_WEIGHT:-0.1}
GEO_TIMESTEP_FRACTION=${GEO_TIMESTEP_FRACTION:-0.5}
BOUNDARY_BLUR_KERNEL=${BOUNDARY_BLUR_KERNEL:-0}
BOUNDARY_BLUR_SIGMA=${BOUNDARY_BLUR_SIGMA:-0.0}
RESOLUTION=${RESOLUTION:-512}
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-6}
CHECKPOINTING_STEPS=${CHECKPOINTING_STEPS:-1000}
CHECKPOINTS_TOTAL_LIMIT=${CHECKPOINTS_TOTAL_LIMIT:-10}
TRACKER_PROJECT_NAME=${TRACKER_PROJECT_NAME:-TokenCompose}
TRACKER_RUN_NAME=${TRACKER_RUN_NAME:-boundary-consistency}
REPORT_TO=${REPORT_TO:-wandb}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
PYTHON_BIN=${PYTHON_BIN:-python}

export CUDA_VISIBLE_DEVICES

##############################
# Derived paths              #
##############################
SPLIT_ROOT="${DATASET_ROOT}/${SPLIT_NAME}"
METADATA_JSONL="${SPLIT_ROOT}/metadata.jsonl"
BOUNDARY_SUBDIR=${BOUNDARY_SUBDIR:-boundary}
BOUNDARY_WIDTH=${BOUNDARY_WIDTH:-3}

mkdir -p "${SPLIT_ROOT}"
mkdir -p "${OUTPUT_DIR}"

#################################
# Stage 1: data preprocessing   #
#################################
(
  cd "${REPO_ROOT}/preprocess_data"
  INPUT_JSON_PATH="${INPUT_JSON}" \
  OUTPUT_JSON_PATH="${METADATA_JSONL}" \
  OUTPUT_DIR="${SPLIT_ROOT}" \
  BOUNDARY_SUBDIR="${BOUNDARY_SUBDIR}" \
  BOUNDARY_WIDTH="${BOUNDARY_WIDTH}" \
  BOUNDARY_BLUR_KERNEL="${BOUNDARY_BLUR_KERNEL}" \
  BOUNDARY_BLUR_SIGMA="${BOUNDARY_BLUR_SIGMA}" \
  PYTHON_BIN="${PYTHON_BIN}" \
    bash run_pipeline.sh
)

echo "Preprocessing complete. Updated metadata written to ${METADATA_JSONL}."

#################################
# Stage 2: model fine-tuning    #
#################################
${PYTHON_BIN} "${REPO_ROOT}/train/src/train_token_compose.py" \
  --pretrained_model_name_or_path="${PRETRAINED_MODEL}" \
  --train_data_dir="${DATASET_ROOT}" \
  --train_batch_size="${TRAIN_BATCH_SIZE}" \
  --resolution="${RESOLUTION}" \
  --dataloader_num_workers="${DATALOADER_NUM_WORKERS}" \
  --gradient_accumulation_steps="${GRADIENT_ACCUMULATION_STEPS}" \
  --gradient_checkpointing \
  --max_train_steps="${MAX_TRAIN_STEPS}" \
  --learning_rate="${LEARNING_RATE}" \
  --max_grad_norm=1 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --output_dir="${OUTPUT_DIR}" \
  --checkpoints_total_limit="${CHECKPOINTS_TOTAL_LIMIT}" \
  --checkpointing_steps="${CHECKPOINTING_STEPS}" \
  --token_loss_scale="${TOKEN_LOSS_SCALE}" \
  --pixel_loss_scale="${PIXEL_LOSS_SCALE}" \
  --train_mid 8 \
  --train_up 16 32 64 \
  --report_to="${REPORT_TO}" \
  --tracker_run_name="${TRACKER_RUN_NAME}" \
  --tracker_project_name="${TRACKER_PROJECT_NAME}" \
  --geo_loss_alpha="${GEO_LOSS_ALPHA}" \
  --geo_boundary_weight="${GEO_BOUNDARY_WEIGHT}" \
  --geo_smooth_weight="${GEO_SMOOTH_WEIGHT}" \
  --geo_timestep_fraction="${GEO_TIMESTEP_FRACTION}" \
  --boundary_blur_kernel="${BOUNDARY_BLUR_KERNEL}" \
  --boundary_blur_sigma="${BOUNDARY_BLUR_SIGMA}"

