
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"


################ params start ##############
INPUT_JSON_PATH="${INPUT_JSON_PATH:-/path/to/input_json.json}"
OUTPUT_JSON_PATH="${OUTPUT_JSON_PATH:-/path/to/output_metadata.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-/path/to/dataset_split_root}"
BOUNDARY_SUBDIR="${BOUNDARY_SUBDIR:-boundary}"
BOUNDARY_WIDTH="${BOUNDARY_WIDTH:-3}"
BOUNDARY_BLUR_KERNEL="${BOUNDARY_BLUR_KERNEL:-0}"
BOUNDARY_BLUR_SIGMA="${BOUNDARY_BLUR_SIGMA:-0.0}"
SAM_CHECKPOINT="${SAM_CHECKPOINT:-/path/to/sam_checkpoint.pth}"
SAM_MODEL_TYPE="${SAM_MODEL_TYPE:-vit_h}"
SAM_POINTS_PER_SIDE="${SAM_POINTS_PER_SIDE:-32}"
SAM_PRED_IOU_THRESH="${SAM_PRED_IOU_THRESH:-0.86}"
SAM_STABILITY_THRESH="${SAM_STABILITY_THRESH:-0.92}"
SAM_MIN_REGION_AREA="${SAM_MIN_REGION_AREA:-256}"
COPY_IMAGES="${COPY_IMAGES:-0}"
IMAGE_SUBDIR="${IMAGE_SUBDIR:-images}"
PYTHON_BIN="${PYTHON_BIN:-python}"
################ params end ##############

mkdir -p "${OUTPUT_DIR}"

prepare_args=(
    --input_json_path "$INPUT_JSON_PATH"
    --dataset_root "$OUTPUT_DIR"
    --output_metadata "$OUTPUT_JSON_PATH"
    --image_subdir "$IMAGE_SUBDIR"
)

if [[ "$COPY_IMAGES" != "0" ]]; then
    prepare_args+=(--copy_images)
fi

# Stage 1: canonicalize captions and align paths with the dataset split.
"${PYTHON_BIN}" prepare_metadata.py "${prepare_args[@]}"

# Stage 2: run SAM automatic mask generator to produce unconditional masks.
"${PYTHON_BIN}" gen_sam_auto_masks.py \
    --input_metadata "$OUTPUT_JSON_PATH" \
    --dataset_root "$OUTPUT_DIR" \
    --output_metadata "$OUTPUT_JSON_PATH" \
    --sam_checkpoint "$SAM_CHECKPOINT" \
    --model_type "$SAM_MODEL_TYPE" \
    --points_per_side "$SAM_POINTS_PER_SIDE" \
    --pred_iou_thresh "$SAM_PRED_IOU_THRESH" \
    --stability_score_thresh "$SAM_STABILITY_THRESH" \
    --min_mask_region_area "$SAM_MIN_REGION_AREA"

################ gen boundary map start #############
"${PYTHON_BIN}" gen_boundary_map.py \
  --input_metadata "$OUTPUT_JSON_PATH" \
  --dataset_root "$OUTPUT_DIR" \
  --output_metadata "$OUTPUT_JSON_PATH" \
  --boundary_subdir "$BOUNDARY_SUBDIR" \
  --boundary_width "$BOUNDARY_WIDTH" \
  --blur_kernel "$BOUNDARY_BLUR_KERNEL" \
  --blur_sigma "$BOUNDARY_BLUR_SIGMA"
################ gen boundary map end #############


################ gen boundary map start #############
"${PYTHON_BIN}" gen_boundary_map.py \
  --input_metadata "$OUTPUT_JSON_PATH" \
  --dataset_root "$OUTPUT_DIR" \
  --output_metadata "$OUTPUT_JSON_PATH" \
  --boundary_subdir "$BOUNDARY_SUBDIR" \
  --boundary_width "$BOUNDARY_WIDTH" \
  --blur_kernel "$BOUNDARY_BLUR_KERNEL" \
  --blur_sigma "$BOUNDARY_BLUR_SIGMA"
################ gen boundary map end #############




