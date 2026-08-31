#!/usr/bin/env python3
"""
Apple Detection Model Evaluation & Validation Script

Evaluates trained model weights on validation or test dataset split,
computing mAP@50, mAP@50-95, Precision, Recall, and inference latency.
"""

import sys
import argparse
from pathlib import Path


def evaluate_model(data_config, model_path, split, imgsz, batch_size, device, project, name):
    from ultralytics import YOLO

    print("📊 Evaluating Apple Detection Model...")
    print(f"📦 Model Weights: {model_path}")
    print(f"📄 Dataset Config: {data_config}")
    print(f"📑 Split: {split}")

    model = YOLO(model_path)

    metrics = model.val(
        data=data_config,
        split=split,
        imgsz=imgsz,
        batch=batch_size,
        device=device,
        project=project,
        name=name,
        plots=True,
        verbose=True
    )

    print("\n" + "=" * 50)
    print("📈 EVALUATION RESULTS SUMMARY")
    print("=" * 50)

    try:
        box_metrics = metrics.box
        print(f"mAP@50     : {box_metrics.map50:.4f}")
        print(f"mAP@50-95  : {box_metrics.map:.4f}")
        print(f"Precision  : {box_metrics.mp:.4f}")
        print(f"Recall     : {box_metrics.mr:.4f}")
    except AttributeError:
        print("Completed validation. Check saved plots in run directory.")

    save_dir = Path(project) / name
    print(f"📁 Detailed plots & confusion matrix saved to: {save_dir.resolve()}")
    print("=" * 50)

    return metrics


def main():
    default_model = "weights/best.pt" if Path("weights/best.pt").exists() else "yolo11n.pt"

    parser = argparse.ArgumentParser(description="Evaluate YOLO Apple Detection Model")
    parser.add_argument("--data", type=str, default="./dataset/apple_dataset/apple_dataset.yaml",
                        help="Path to dataset YAML file")
    parser.add_argument("--model", type=str, default=default_model,
                        help="Path to model weights file (.pt)")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"],
                        help="Dataset split to evaluate ('val' or 'test')")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--device", type=str, default="", help="Device ('mps', 'cuda', 'cpu')")
    parser.add_argument("--project", type=str, default="runs/val", help="Evaluation output project path")
    parser.add_argument("--name", type=str, default="apple_eval", help="Experiment name")

    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"❌ Error: Dataset config '{args.data}' not found!")
        print("Please run `python3 convert_coco_to_yolo.py` first.")
        sys.exit(1)

    evaluate_model(
        data_config=str(data_path),
        model_path=args.model,
        split=args.split,
        imgsz=args.imgsz,
        batch_size=args.batch,
        device=args.device,
        project=args.project,
        name=args.name
    )


if __name__ == "__main__":
    main()
