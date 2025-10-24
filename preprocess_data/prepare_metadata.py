import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List


def _normalize_caption(raw_caption: Any) -> str:
    if isinstance(raw_caption, str):
        return raw_caption
    if isinstance(raw_caption, Iterable):
        # Pick the first caption when multiple candidates are provided.
        for candidate in raw_caption:
            if isinstance(candidate, str):
                return candidate
    raise ValueError("Caption must be a string or a list/tuple of strings.")


def _prepare_entry(
    entry: Dict[str, Any],
    dataset_root: Path,
    copy_images: bool,
    image_subdir: Path,
) -> Dict[str, Any]:
    image_path = Path(entry["img_path"]).expanduser().resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Image path {image_path} not found.")

    caption = _normalize_caption(entry["caption"])

    if copy_images:
        target_dir = (dataset_root / image_subdir).resolve()
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / image_path.name
        if target_path.resolve() != image_path.resolve():
            shutil.copy2(image_path, target_path)
        relative_path = target_path.relative_to(dataset_root)
    else:
        try:
            relative_path = image_path.relative_to(dataset_root)
        except ValueError as exc:
            raise ValueError(
                "Image path lies outside dataset_root. Either move your images under the "
                "dataset root or rerun with --copy-images enabled."
            ) from exc

    return {
        "file_name": relative_path.as_posix(),
        "text": caption,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare metadata.jsonl for boundary supervision training.")
    parser.add_argument("--input_json_path", type=str, required=True, help="Input JSON/JSONL file with img_path and caption fields.")
    parser.add_argument("--dataset_root", type=str, required=True, help="Root directory that will host the ImageFolder split.")
    parser.add_argument("--output_metadata", type=str, required=True, help="Output metadata.jsonl path.")
    parser.add_argument("--copy_images", action="store_true", help="Copy source images into dataset_root/image_subdir.")
    parser.add_argument("--image_subdir", type=str, default="images", help="Sub-directory used when --copy_images is enabled.")
    return parser.parse_args()


def main():
    args = parse_args()

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    dataset_root.mkdir(parents=True, exist_ok=True)

    input_path = Path(args.input_json_path).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input metadata {input_path} not found.")

    with open(input_path, "r") as f:
        raw_text = f.read().strip()

    if raw_text.startswith("["):
        raw_entries: List[Dict[str, Any]] = json.loads(raw_text)
    else:
        raw_entries = [json.loads(line) for line in raw_text.splitlines() if line.strip()]

    processed_entries = []
    for entry in raw_entries:
        processed_entries.append(
            _prepare_entry(
                entry=entry,
                dataset_root=dataset_root,
                copy_images=args.copy_images,
                image_subdir=Path(args.image_subdir),
            )
        )

    output_path = Path(args.output_metadata).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for entry in processed_entries:
            f.write(json.dumps(entry) + "\n")


if __name__ == "__main__":
    main()
