
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"


################ params start ##############
INPUT_JSON_PATH="${INPUT_JSON_PATH:-/path/to/input_json.json}"
OUTPUT_JSON_PATH="${OUTPUT_JSON_PATH:-/path/to/output_json.json}"
OUTPUT_DIR="${OUTPUT_DIR:-/path/to/segmentation_output_dir}"
BOUNDARY_SUBDIR="${BOUNDARY_SUBDIR:-boundary}"
BOUNDARY_WIDTH="${BOUNDARY_WIDTH:-3}"
BOUNDARY_BLUR_KERNEL="${BOUNDARY_BLUR_KERNEL:-0}"
BOUNDARY_BLUR_SIGMA="${BOUNDARY_BLUR_SIGMA:-0.0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
################ params end ##############

mkdir -p "${OUTPUT_DIR}"

################ gen sentence noun tags start #############
"${PYTHON_BIN}" gen_noun_tgt.py \
    --input_json_path "$INPUT_JSON_PATH" \
    --output_json_path "$OUTPUT_JSON_PATH"
################ gen sentence noun tags end #############

################ gen mask start #############
"${PYTHON_BIN}" gen_mask.py \
  --config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
  --grounded_checkpoint model_ckpt/groundingdino_swint_ogc.pth \
  --sam_hq_checkpoint model_ckpt/sam_hq_vit_h.pth \
  --use_sam_hq \
  --output_dir "$OUTPUT_DIR" \
  --output_jsonl "$OUTPUT_JSON_PATH" \
  --input_metadata "$OUTPUT_JSON_PATH" \
  --box_threshold 0.25 \
  --text_threshold 0.25 \
  --device "cuda"
################ gen mask end #############


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




