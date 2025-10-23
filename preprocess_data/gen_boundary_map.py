import argparse
import json
import os
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm


def mask_to_boundary(mask: torch.Tensor, width: int) -> torch.Tensor:
    width = max(int(width), 1)
    kernel_size = width * 2 + 1
    padding = width

    mask = mask.float().unsqueeze(0).unsqueeze(0)
    dilated = F.max_pool2d(mask, kernel_size=kernel_size, stride=1, padding=padding)
    eroded = 1.0 - F.max_pool2d(1.0 - mask, kernel_size=kernel_size, stride=1, padding=padding)
    boundary = (dilated - eroded).clamp(min=0.0)
    return boundary.squeeze(0).squeeze(0)


def gaussian_blur(tensor: torch.Tensor, kernel_size: int, sigma: float) -> torch.Tensor:
    if kernel_size <= 0 or sigma <= 0:
        return tensor
    if kernel_size % 2 == 0:
        kernel_size += 1

    coords = torch.arange(kernel_size, dtype=tensor.dtype, device=tensor.device)
    coords = coords - (kernel_size - 1) / 2
    kernel_1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = torch.matmul(kernel_1d.unsqueeze(1), kernel_1d.unsqueeze(0))
    kernel_2d = kernel_2d / kernel_2d.sum()
    kernel_2d = kernel_2d.view(1, 1, kernel_size, kernel_size)

    padding = kernel_size // 2
    return F.conv2d(tensor, kernel_2d, padding=padding)


def load_binary_mask(path: Path) -> torch.Tensor:
    mask = Image.open(path).convert("L")
    mask_array = np.array(mask, dtype=np.float32)
    mask_tensor = torch.from_numpy(mask_array)
    mask_tensor = (mask_tensor > 0.0).float()
    return mask_tensor


def build_boundary_map(
    mask_paths: List[Path],
    boundary_width: int,
    blur_kernel: int,
    blur_sigma: float,
    fallback_size: torch.Size,
) -> torch.Tensor:
    boundary_components = []
    for mask_path in mask_paths:
        mask_tensor = load_binary_mask(mask_path)
        boundary_components.append(mask_to_boundary(mask_tensor, boundary_width))

    if not boundary_components:
        height, width = fallback_size
        return torch.zeros((height, width), dtype=torch.float32)

    stacked = torch.stack(boundary_components, dim=0).sum(dim=0)
    stacked = stacked.clamp(min=0.0)
    if stacked.max() > 0:
        stacked = stacked / stacked.max()

    stacked = stacked.unsqueeze(0).unsqueeze(0)
    stacked = gaussian_blur(stacked, blur_kernel, blur_sigma)
    return stacked.squeeze(0).squeeze(0).clamp(0.0, 1.0)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate normalized boundary maps from segmentation masks.")
    parser.add_argument("--input_metadata", type=str, required=True, help="Path to metadata.jsonl produced by the segmentation pipeline.")
    parser.add_argument("--dataset_root", type=str, required=True, help="Root directory that contains the images and segmentation outputs.")
    parser.add_argument("--output_metadata", type=str, required=True, help="Where to write the updated metadata with boundary paths.")
    parser.add_argument("--boundary_subdir", type=str, default="boundary", help="Relative sub-directory (under dataset_root) to store the generated boundary maps.")
    parser.add_argument("--boundary_width", type=int, default=3, help="Width of the boundary band in pixels.")
    parser.add_argument("--blur_kernel", type=int, default=0, help="Kernel size for optional Gaussian smoothing (0 disables blur).")
    parser.add_argument("--blur_sigma", type=float, default=0.0, help="Sigma for optional Gaussian smoothing (0 disables blur).")
    return parser.parse_args()


def main():
    args = parse_args()

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    input_metadata = Path(args.input_metadata).expanduser().resolve()
    output_metadata = Path(args.output_metadata).expanduser().resolve()
    boundary_root = dataset_root / args.boundary_subdir
    boundary_root.mkdir(parents=True, exist_ok=True)

    with open(input_metadata, "r") as f:
        entries = [json.loads(line) for line in f]

    updated_entries = []
    for entry in tqdm(entries, desc="Boundary maps"):
        attn_list = entry.get("attn_list", [])
        mask_paths = []
        for _, rel_path in attn_list:
            if rel_path is None:
                continue
            mask_path = dataset_root / rel_path
            if not mask_path.exists():
                raise FileNotFoundError(f"Mask path {mask_path} not found.")
            mask_paths.append(mask_path)

        image_path = dataset_root / entry["file_name"]
        if not image_path.exists():
            raise FileNotFoundError(f"Image path {image_path} not found.")
        with Image.open(image_path) as img:
            fallback_size = torch.Size([img.height, img.width])

        boundary_tensor = build_boundary_map(
            mask_paths=mask_paths,
            boundary_width=args.boundary_width,
            blur_kernel=args.blur_kernel,
            blur_sigma=args.blur_sigma,
            fallback_size=fallback_size,
        )

        boundary_filename = f"{Path(entry['file_name']).stem}_boundary.png"
        boundary_path = boundary_root / boundary_filename
        boundary_image = Image.fromarray((boundary_tensor.cpu().numpy() * 255.0).astype(np.uint8))
        boundary_image.save(boundary_path)

        entry["boundary_path"] = os.path.relpath(boundary_path, dataset_root)
        updated_entries.append(entry)

    with open(output_metadata, "w") as f:
        for entry in updated_entries:
            f.write(json.dumps(entry) + "\n")
if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()

