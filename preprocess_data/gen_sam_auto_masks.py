import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry


def parse_args():
    parser = argparse.ArgumentParser(description="Run SAM automatic mask generator and store masks for each image.")
    parser.add_argument("--input_metadata", type=str, required=True, help="metadata.jsonl produced by prepare_metadata.py")
    parser.add_argument("--dataset_root", type=str, required=True, help="Root directory containing the dataset split")
    parser.add_argument("--output_metadata", type=str, required=True, help="Where to store the updated metadata with mask paths")
    parser.add_argument("--sam_checkpoint", type=str, required=True, help="Path to the SAM/SAM-HQ checkpoint")
    parser.add_argument("--model_type", type=str, default="vit_h", help="SAM model type (vit_h, vit_l, vit_b, ...)")
    parser.add_argument("--output_subdir", type=str, default="seg", help="Sub-directory to store generated masks")
    parser.add_argument("--points_per_side", type=int, default=32)
    parser.add_argument("--pred_iou_thresh", type=float, default=0.86)
    parser.add_argument("--stability_score_thresh", type=float, default=0.92)
    parser.add_argument("--min_mask_region_area", type=int, default=256, help="Filter out masks smaller than this area")
    return parser.parse_args()


def main():
    args = parse_args()

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    metadata_path = Path(args.input_metadata).expanduser().resolve()
    output_path = Path(args.output_metadata).expanduser().resolve()
    mask_root = dataset_root / args.output_subdir
    mask_root.mkdir(parents=True, exist_ok=True)

    with open(metadata_path, "r") as f:
        entries: List[dict] = [json.loads(line) for line in f]

    sam = sam_model_registry[args.model_type](checkpoint=args.sam_checkpoint)
    sam.to(device="cuda" if torch.cuda.is_available() else "cpu")
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=args.points_per_side,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_score_thresh,
        min_mask_region_area=args.min_mask_region_area,
    )

    updated_entries = []
    for entry in tqdm(entries, desc="SAM masks"):
        image_path = dataset_root / entry["file_name"]
        if not image_path.exists():
            raise FileNotFoundError(f"Image path {image_path} not found.")

        image = np.array(Image.open(image_path).convert("RGB"))
        masks = mask_generator.generate(image)

        image_stem = Path(entry["file_name"]).stem
        image_mask_dir = mask_root / image_stem
        image_mask_dir.mkdir(parents=True, exist_ok=True)

        mask_paths = []
        for idx, mask in enumerate(masks):
            mask_array = mask["segmentation"].astype(np.uint8) * 255
            mask_path = image_mask_dir / f"{image_stem}_mask_{idx:03d}.png"
            Image.fromarray(mask_array).save(mask_path)
            mask_paths.append(os.path.relpath(mask_path, dataset_root))

        entry["mask_paths"] = mask_paths
        updated_entries.append(entry)

    with open(output_path, "w") as f:
        for entry in updated_entries:
            f.write(json.dumps(entry) + "\n")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
