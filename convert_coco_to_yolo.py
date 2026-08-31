#!/usr/bin/env python3
"""
COCO to YOLO Format Converter for Apple Detection Dataset.

Converts Roboflow COCO format JSON annotations into Ultralytics YOLO format,
organizing files into images/ and labels/ directories for train, valid, and test splits,
and generates apple_dataset.yaml.
"""

import os
import sys
import json
import shutil
import argparse
from pathlib import Path

# Configure stdout to handle utf-8 on Windows command prompts
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass


def convert_bbox_coco_to_yolo(coco_bbox, img_width, img_height):
    """
    Converts COCO bbox [x_min, y_min, w, h] to YOLO [x_center, y_center, w, h] (normalized).
    """
    x_min, y_min, w, h = coco_bbox
    if img_width <= 0 or img_height <= 0:
        return None

    x_center = x_min + w / 2.0
    y_center = y_min + h / 2.0

    # Normalize by image dimensions
    x_center_norm = max(0.0, min(1.0, x_center / img_width))
    y_center_norm = max(0.0, min(1.0, y_center / img_height))
    w_norm = max(0.0, min(1.0, w / img_width))
    h_norm = max(0.0, min(1.0, h / img_height))

    return x_center_norm, y_center_norm, w_norm, h_norm


def convert_dataset(coco_dir, output_dir):
    coco_path = Path(coco_dir).resolve()
    out_path = Path(output_dir).resolve()

    if not coco_path.exists():
        raise FileNotFoundError(f"Source COCO directory not found: {coco_path}")

    print(f"📦 Converting COCO dataset from: {coco_path}")
    print(f"📂 Target YOLO directory: {out_path}")

    splits = ['train', 'valid', 'test']
    all_categories = {}
    cat_id_to_yolo_id = {}

    # First pass: collect categories across available splits
    for split in splits:
        split_dir = coco_path / split
        anno_file = split_dir / "_annotations.coco.json"
        if anno_file.exists():
            with open(anno_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for cat in data.get('categories', []):
                    cname = cat['name'].strip().lower()
                    if cname not in all_categories:
                        all_categories[cname] = len(all_categories)
                    cat_id_to_yolo_id[(split, cat['id'])] = all_categories[cname]

    if not all_categories:
        # Fallback to single class 'apple' if no categories found
        all_categories['apple'] = 0

    class_names = [k for k, v in sorted(all_categories.items(), key=lambda item: item[1])]
    print(f"🏷️  Discovered categories ({len(class_names)}): {class_names}")

    # Process each split
    for split in splits:
        split_dir = coco_path / split
        anno_file = split_dir / "_annotations.coco.json"

        if not split_dir.exists() or not anno_file.exists():
            print(f"⚠️ Split '{split}' missing or _annotations.coco.json not found. Skipping.")
            continue

        print(f"\n🔄 Processing split: '{split}'...")
        with open(anno_file, 'r', encoding='utf-8') as f:
            coco_data = json.load(f)

        img_out_dir = out_path / "images" / split
        lbl_out_dir = out_path / "labels" / split
        img_out_dir.mkdir(parents=True, exist_ok=True)
        lbl_out_dir.mkdir(parents=True, exist_ok=True)

        images = {img['id']: img for img in coco_data.get('images', [])}
        annotations = coco_data.get('annotations', [])

        # Group annotations by image_id
        img_annos = {}
        for anno in annotations:
            img_id = anno['image_id']
            img_annos.setdefault(img_id, []).append(anno)

        converted_count = 0
        bbox_count = 0

        for img_id, img_info in images.items():
            filename = img_info['file_name']
            src_img_path = split_dir / filename

            if not src_img_path.exists():
                # Check for unescaped / converted filename
                filename_clean = filename.split('/')[-1]
                src_img_path = split_dir / filename_clean
                if not src_img_path.exists():
                    continue

            # Copy image file to YOLO images directory
            dst_img_path = img_out_dir / src_img_path.name
            shutil.copy2(src_img_path, dst_img_path)

            # Create corresponding label file
            txt_name = src_img_path.stem + ".txt"
            dst_lbl_path = lbl_out_dir / txt_name

            img_w = img_info.get('width', 0)
            img_h = img_info.get('height', 0)

            label_lines = []
            for anno in img_annos.get(img_id, []):
                cat_id = anno['category_id']
                yolo_cls = cat_id_to_yolo_id.get((split, cat_id), 0)

                bbox = anno.get('bbox', [])
                if len(bbox) == 4 and img_w > 0 and img_h > 0:
                    yolo_bbox = convert_bbox_coco_to_yolo(bbox, img_w, img_h)
                    if yolo_bbox is not None:
                        xc, yc, w, h = yolo_bbox
                        label_lines.append(f"{yolo_cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
                        bbox_count += 1

            with open(dst_lbl_path, 'w', encoding='utf-8') as lf:
                lf.write("\n".join(label_lines) + ("\n" if label_lines else ""))

            converted_count += 1

        print(f"✅ Split '{split}' complete: {converted_count} images, {bbox_count} bounding boxes.")

    # Create dataset yaml file dynamically based on output directory name
    yaml_name = out_path.name
    if yaml_name.endswith("_yolo"):
        yaml_name = yaml_name[:-5] # remove '_yolo' suffix if present
    yaml_path = out_path / f"{yaml_name}.yaml"
    with open(yaml_path, 'w', encoding='utf-8') as yf:
        yf.write(f"path: {out_path.as_posix()}\n")
        yf.write(f"train: images/train\n")
        yf.write(f"val: images/valid\n")
        yf.write(f"test: images/test\n\n")
        yf.write(f"names:\n")
        for idx, name in enumerate(class_names):
            yf.write(f"  {idx}: {name}\n")

    print(f"\n🎉 Conversion successfully finished!")
    print(f"📄 Created dataset config: {yaml_path}")
    return yaml_path


def main():
    parser = argparse.ArgumentParser(description="Convert COCO Apple dataset to YOLO format")
    parser.add_argument("--coco-dir", type=str, default="/Volumes/MacHDD/Shinjuku/apple-Forked on 8-25-2026.coco",
                        help="Path to root directory of COCO dataset")
    parser.add_argument("--output-dir", type=str, default="./apple_dataset",
                        help="Path to output YOLO dataset directory")
    args = parser.parse_args()

    convert_dataset(args.coco_dir, args.output_dir)


if __name__ == "__main__":
    main()
