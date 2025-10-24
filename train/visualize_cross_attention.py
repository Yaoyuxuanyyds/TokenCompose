#!/usr/bin/env python3
"""Utility to visualize Stable Diffusion cross-attention maps for a given token."""

import argparse
import os
import sys
from typing import List, Optional

import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusionPipeline

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(SCRIPT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from attn_utils import AttentionStore, register_attention_control, get_cross_attn_map_from_unet  # noqa: E402
from loss_utils import get_word_idx  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a Stable Diffusion pipeline and export the cross-attention heatmap "
            "for the specified token."
        )
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        default="runwayml/stable-diffusion-v1-5",
        help="Path or identifier of the (fine-tuned) Stable Diffusion checkpoint.",
    )
    parser.add_argument("--prompt", required=True, help="Text prompt used for generation.")
    parser.add_argument(
        "--token",
        required=True,
        help="Token (or whitespace separated phrase) whose attention map will be visualized.",
    )
    parser.add_argument(
        "--layer",
        default="mid_32",
        help=(
            "Attention layer key to visualize (e.g. down_32, mid_32, up_16)."
            " Keys will be printed when the script finishes."
        ),
    )
    parser.add_argument(
        "--output_dir",
        default="visualizations",
        help="Directory where the generated image and attention map will be stored.",
    )
    parser.add_argument(
        "--output_prefix",
        default=None,
        help="Optional prefix for the saved files. Defaults to the sanitized token name.",
    )
    parser.add_argument(
        "--negative_prompt",
        default=None,
        help="Optional negative prompt for classifier-free guidance.",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=30,
        help="Number of denoising steps used by the pipeline.",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="Classifier-free guidance scale applied during sampling.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (uses CUDA generator when available).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Optional override for output image height (defaults to model configuration).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Optional override for output image width (defaults to model configuration).",
    )
    parser.add_argument(
        "--overlay_alpha",
        type=float,
        default=0.6,
        help="Alpha multiplier applied to the heatmap overlay when blending with the image.",
    )
    parser.add_argument(
        "--is_sd21",
        action="store_true",
        help="Indicate that the checkpoint uses the 2.x architecture (for resolution scaling).",
    )
    parser.add_argument(
        "--half_precision",
        action="store_true",
        help="Run the pipeline in float16 when a CUDA device is available.",
    )
    parser.add_argument(
        "--disable_safety_checker",
        action="store_true",
        help="Disable the safety checker for research scenarios.",
    )
    parser.add_argument(
        "--save_numpy",
        action="store_true",
        help="Store the raw attention map as a NumPy array alongside the visualization.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Manually select the torch device (defaults to cuda if available, else cpu).",
    )
    return parser.parse_args()


def _prepare_pipeline(
    model_path: str,
    device: torch.device,
    half_precision: bool,
    disable_safety_checker: bool,
) -> StableDiffusionPipeline:
    dtype = torch.float16 if half_precision and device.type == "cuda" else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=dtype)
    if disable_safety_checker:
        pipe.safety_checker = None
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()
    return pipe


def _sanitize_token(token: str) -> str:
    sanitized = token.strip().lower().replace(" ", "_")
    sanitized = "".join(ch for ch in sanitized if ch.isalnum() or ch in {"_", "-"})
    return sanitized or "token"


def _aggregate_attention(
    attn_maps: List[torch.Tensor], token_indices: List[int]
) -> torch.Tensor:
    if len(attn_maps) == 0:
        raise ValueError("No attention maps were collected for the requested layer.")
    stacked = torch.stack(attn_maps, dim=0).float()
    mean_over_layers = stacked.mean(dim=0)
    mean_over_heads = mean_over_layers.mean(dim=0)
    token_map = mean_over_heads[..., token_indices].mean(dim=-1)
    return token_map


def _normalize_map(attn_map: torch.Tensor) -> torch.Tensor:
    attn_map = attn_map - attn_map.min()
    max_val = attn_map.max()
    if max_val > 0:
        attn_map = attn_map / max_val
    return attn_map


def _save_outputs(
    image: Image.Image,
    attn_map: torch.Tensor,
    output_dir: str,
    prefix: str,
    layer: str,
    overlay_alpha: float,
    save_numpy: bool,
):
    os.makedirs(output_dir, exist_ok=True)
    base_path = os.path.join(output_dir, prefix)

    image_path = f"{base_path}_image.png"
    heatmap_path = f"{base_path}_{layer}_heatmap.png"
    overlay_path = f"{base_path}_{layer}_overlay.png"
    numpy_path = f"{base_path}_{layer}_heatmap.npy"

    image.save(image_path)

    attn_np = attn_map.detach().cpu().numpy()
    attn_img = Image.fromarray(np.uint8(np.clip(attn_np * 255.0, 0, 255)), mode="L")
    attn_img = attn_img.resize(image.size, resample=Image.BICUBIC)
    attn_img.save(heatmap_path)

    attn_resized = np.array(attn_img, dtype=np.float32) / 255.0
    overlay = np.zeros((attn_resized.shape[0], attn_resized.shape[1], 4), dtype=np.uint8)
    overlay[..., 0] = 255
    overlay[..., 1] = 64
    overlay[..., 2] = 0
    overlay[..., 3] = np.clip(attn_resized * overlay_alpha * 255.0, 0, 255).astype(np.uint8)
    overlay_img = Image.fromarray(overlay, mode="RGBA")

    blended = Image.alpha_composite(image.convert("RGBA"), overlay_img)
    blended.save(overlay_path)

    if save_numpy:
        np.save(numpy_path, attn_np)

    return {
        "image": image_path,
        "heatmap": heatmap_path,
        "overlay": overlay_path,
        "numpy": numpy_path if save_numpy else None,
    }


def main():
    args = parse_args()

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    generator: Optional[torch.Generator] = None
    if args.seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(args.seed)

    pipe = _prepare_pipeline(
        model_path=args.pretrained_model_name_or_path,
        device=device,
        half_precision=args.half_precision,
        disable_safety_checker=args.disable_safety_checker,
    )

    controller = AttentionStore()
    register_attention_control(pipe.unet, controller)
    controller.reset()

    extra_kwargs = {}
    if args.height is not None:
        extra_kwargs["height"] = args.height
    if args.width is not None:
        extra_kwargs["width"] = args.width

    with torch.autocast(device.type, enabled=(device.type == "cuda" and args.half_precision)):
        with torch.inference_mode():
            output = pipe(
                prompt=args.prompt,
                negative_prompt=args.negative_prompt,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps,
                generator=generator,
                **extra_kwargs,
            )
    image = output.images[0]

    if hasattr(output, "nsfw_content_detected") and output.nsfw_content_detected:
        if any(output.nsfw_content_detected):
            print("[warning] NSFW content was detected in the generated image.")

    attn_dict = get_cross_attn_map_from_unet(
        attention_store=controller, is_training_sd21=args.is_sd21
    )

    available_layers = sorted(attn_dict.keys())
    if args.layer not in attn_dict:
        raise ValueError(
            f"Requested layer '{args.layer}' not found. Available options: {available_layers}"
        )

    token_indices = get_word_idx(args.prompt, args.token, pipe.tokenizer)
    attn_map = _aggregate_attention(attn_dict[args.layer], token_indices)
    attn_map = _normalize_map(attn_map)

    prefix = args.output_prefix or _sanitize_token(args.token)
    saved_paths = _save_outputs(
        image=image,
        attn_map=attn_map,
        output_dir=args.output_dir,
        prefix=prefix,
        layer=args.layer,
        overlay_alpha=args.overlay_alpha,
        save_numpy=args.save_numpy,
    )

    print("Saved files:")
    for label, path in saved_paths.items():
        if path is not None:
            print(f"  {label:>7}: {path}")
    print("Available layers:", ", ".join(available_layers))
    print("Token indices:", token_indices)


if __name__ == "__main__":
    main()
