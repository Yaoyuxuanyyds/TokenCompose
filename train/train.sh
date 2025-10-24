#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

MODEL_NAME="${MODEL_NAME:-CompVis/stable-diffusion-v1-4}"
IMAGE_RESOLUTION="${IMAGE_RESOLUTION:-512}"

# path to training dataset
TRAIN_DIR="${TRAIN_DIR:-data/coco_gsam_img}"

# set up wandb project
PROJ_NAME="${PROJ_NAME:-TokenCompose}"
RUN_NAME="${RUN_NAME:-TokenCompose}"

# checkpoint settings
CHECKPOINT_STEP="${CHECKPOINT_STEP:-8000}"
CHECKPOINT_LIMIT="${CHECKPOINT_LIMIT:-10}"

# allow 500 extra steps to be safe
MAX_TRAINING_STEPS="${MAX_TRAINING_STEPS:-24500}"

# loss and lr settings (the TokenCompose token/pixel objectives are disabled; only
# denoising + boundary regularisation remain)
LEARNING_RATE="${LEARNING_RATE:-5e-6}"
GEO_LOSS_ALPHA="${GEO_LOSS_ALPHA:-0.2}"
GEO_BOUNDARY_WEIGHT="${GEO_BOUNDARY_WEIGHT:-1.0}"
GEO_SMOOTH_WEIGHT="${GEO_SMOOTH_WEIGHT:-0.1}"
GEO_TIMESTEP_FRACTION="${GEO_TIMESTEP_FRACTION:-0.5}"
BOUNDARY_BLUR_KERNEL="${BOUNDARY_BLUR_KERNEL:-0}"
BOUNDARY_BLUR_SIGMA="${BOUNDARY_BLUR_SIGMA:-0.0}"

# other settings
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-6}"

OUTPUT_DIR="${OUTPUT_DIR:-results/${RUN_NAME}}"

mkdir -p "$OUTPUT_DIR"

python src/train_token_compose.py \
  --pretrained_model_name_or_path="$MODEL_NAME" \
  --train_data_dir="$TRAIN_DIR" \
  --train_batch_size=1 \
  --resolution "$IMAGE_RESOLUTION" \
  --dataloader_num_workers "$DATALOADER_NUM_WORKERS" \
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
  --gradient_checkpointing \
  --max_train_steps="$MAX_TRAINING_STEPS" \
  --learning_rate "$LEARNING_RATE" \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir="$OUTPUT_DIR" \
  --checkpoints_total_limit "$CHECKPOINT_LIMIT" \
  --checkpointing_steps "$CHECKPOINT_STEP" \
  --train_mid 8 \
  --train_up 16 32 64 \
  --report_to="wandb" \
  --tracker_run_name "$RUN_NAME" \
  --tracker_project_name "$PROJ_NAME" \
  --geo_loss_alpha "$GEO_LOSS_ALPHA" \
  --geo_boundary_weight "$GEO_BOUNDARY_WEIGHT" \
  --geo_smooth_weight "$GEO_SMOOTH_WEIGHT" \
  --geo_timestep_fraction "$GEO_TIMESTEP_FRACTION" \
  --boundary_blur_kernel "$BOUNDARY_BLUR_KERNEL" \
  --boundary_blur_sigma "$BOUNDARY_BLUR_SIGMA"
