import json
import argparse
from pathlib import Path
from PIL import Image


def is_small_image(image_path: str) -> bool:
    """Check if the image is smaller than 28x28."""
    try:
        with Image.open(image_path) as img:
            w, h = img.size
            return w < 28 or h < 28
    except Exception as e:
        # if image can't be opened, treat it as invalid
        print(f"Warning: could not open {image_path}, skipping. ({e})")
        return True


def main(input_path: str, output_path: str):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    total_before = len(data)
    count_rethink = 0
    count_smallimg = 0
    filtered_data = []
    examples = []

    for item in data:
        keep = True

        # --- rule 1: filter "let me rethink it" ---
        for msg in item.get("messages", []):
            if msg.get("role") == "assistant" and "let me rethink it" in msg.get("content", "").lower():
                count_rethink += 1
                keep = False
                if len(examples) < 3:
                    examples.append({"reason": "rethink", "sample": item})
                break

        # --- rule 2: filter small images ---
        if keep:
            images = item.get("images", [])
            if isinstance(images, str):
                images = [images]
            for img_path in images:
                if is_small_image(img_path):
                    count_smallimg += 1
                    keep = False
                    if len(examples) < 3:
                        examples.append({"reason": "small_image", "sample": item})
                    break

        if keep:
            filtered_data.append(item)

    total_after = len(filtered_data)

    # save new file
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(filtered_data, f, ensure_ascii=False, indent=2)

    print(f"Samples before filtering: {total_before}")
    print(f"Removed {count_rethink} samples containing 'let me rethink it'")
    print(f"Removed {count_smallimg} samples with image size < 28")
    print(f"Samples after filtering: {total_after}")
    print(f"Saved to: {output_path}")

    if examples:
        print("Some removed examples:")
        for ex in examples:
            print(f"Reason: {ex['reason']}")
            print(json.dumps(ex["sample"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter JSON dataset")
    parser.add_argument("--input", type=str, required=True, help="Path to input JSON file")
    parser.add_argument("--output", type=str, required=True, help="Path to save filtered JSON file")
    args = parser.parse_args()

    main(args.input, args.output)
