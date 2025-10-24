## Preprocess Data Pipeline

### Overview

The updated pipeline is aligned with the attention-boundary supervision objective and no longer relies on token-specific masks. Given an `input.json` containing entries such as

```
[
    {
        "img_path": "/absolute/path/to/image1.jpg",
        "caption": ["caption option 1", "caption option 2"]
    },
    ...
]
```

`run_pipeline.sh` performs the following steps:

1. **prepare_metadata.py** converts the raw JSON into `metadata.jsonl`, normalises captions, and ensures every `file_name` is rooted inside the dataset split.
2. **gen_sam_auto_masks.py** runs the [Segment Anything](https://github.com/facebookresearch/segment-anything) automatic mask generator (SAM or SAM-HQ checkpoints) to obtain unconditional instance masks for every image.
3. **gen_boundary_map.py** aggregates the masks into a soft boundary map that matches the cross-attention spatial resolution expected by the boundary-consistency loss.

The resulting `metadata.jsonl` contains exactly the fields consumed by the new training code: `file_name`, `text`, `mask_paths`, and `boundary_path`.

### Setup Environment

```bash
conda create -n preprocess_data python=3.10
conda activate preprocess_data
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install git+https://github.com/facebookresearch/segment-anything.git
pip install pillow tqdm numpy
```

Download a SAM/SAM-HQ checkpoint (for example `sam_vit_h_4b8939.pth`) into `preprocess_data/model_ckpt/` or any location you prefer.

### Run the pipeline

Export or edit the variables consumed by `run_pipeline.sh`:

```bash
INPUT_JSON_PATH=/path/to/input.json
OUTPUT_JSON_PATH=/path/to/train/metadata.jsonl
OUTPUT_DIR=/path/to/train
SAM_CHECKPOINT=/path/to/sam_vit_h_4b8939.pth
SAM_MODEL_TYPE=vit_h   # vit_l, vit_b, or any key supported by segment_anything
BOUNDARY_SUBDIR=boundary
BOUNDARY_WIDTH=3
BOUNDARY_BLUR_KERNEL=0
BOUNDARY_BLUR_SIGMA=0.0
# Optional overrides:
#   COPY_IMAGES=0     # reuse images in-place instead of copying into OUTPUT_DIR/images
#   IMAGE_SUBDIR=imgs # customise the copy destination when COPY_IMAGES=1
```

Then launch the preprocessing pipeline:

```bash
cd preprocess_data
bash run_pipeline.sh
```

The generated assets follow the structure below:

```
OUTPUT_DIR/
    images/              # present only when --copy-images is passed to prepare_metadata.py
    seg/
        <image_stem>/
            <image_stem>_mask_000.png
            <image_stem>_mask_001.png
            ...
    boundary/
        <image_stem>_boundary.png
    metadata.jsonl
```

Each entry inside `metadata.jsonl` stores relative paths so the HuggingFace `imagefolder` loader can read the dataset directly.

> 💡 Tip: the repository root also contains [`run_full_pipeline.sh`](../run_full_pipeline.sh), which chains this preprocessing stage together with `train/src/train_token_compose.py` so you can launch the entire training job with a single command.
