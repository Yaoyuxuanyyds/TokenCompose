import argparse
import json
from pathlib import Path

from tqdm import tqdm
from transformers import CLIPTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Validate that every caption in metadata.jsonl can be tokenised by CLIP.")
    parser.add_argument("--metadata", type=str, required=True, help="Path to metadata.jsonl produced by the preprocessing pipeline.")
    parser.add_argument("--pretrained_model", type=str, default="CompVis/stable-diffusion-v1-4", help="Base Stable Diffusion model whose tokenizer will be used for validation.")
    return parser.parse_args()


def main():
    args = parse_args()
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model, subfolder="tokenizer")

    metadata_path = Path(args.metadata).expanduser().resolve()
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file {metadata_path} not found.")

    with open(metadata_path, "r") as f:
        entries = [json.loads(line) for line in f if line.strip()]

    failures = []
    for entry in tqdm(entries, desc="Tokenising"):
        text = entry.get("text")
        if not isinstance(text, str):
            failures.append((entry.get("file_name"), "missing text"))
            continue

        try:
            tokenizer(text)
        except Exception as exc:  # noqa: BLE001
            failures.append((entry.get("file_name"), str(exc)))

    if failures:
        print("Found captions that failed to tokenize:")
        for file_name, error in failures:
            print(f"  {file_name}: {error}")
    else:
        print("All captions successfully tokenized.")


if __name__ == "__main__":
    main()
