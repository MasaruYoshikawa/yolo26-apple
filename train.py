#!/usr/bin/env python3
"""
YOLO Apple Detection Model Trainer

Fine-tunes a YOLO model on the apple dataset and saves the best model weights.
"""

import os
import sys
import shutil
import argparse
import contextlib
from pathlib import Path

# Enable PyTorch MPS fallback for operations not natively implemented in MPS C++
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Monkey-patch PyTorch and Ultralytics autocast to prevent 'unsupported autocast device_type mps' crash on Apple Silicon
try:
    import torch
    import ultralytics.utils.torch_utils
    
    _orig_torch_autocast = torch.amp.autocast
    class SafeTorchAutocast(_orig_torch_autocast):
        def __init__(self, device_type, dtype=None, enabled=True, cache_enabled=None):
            if device_type == "mps":
                device_type = "cpu"
                enabled = False
            super().__init__(device_type, dtype=dtype, enabled=enabled, cache_enabled=cache_enabled)
    torch.amp.autocast = SafeTorchAutocast

    _orig_ultralytics_autocast = ultralytics.utils.torch_utils.autocast
    def safe_ultralytics_autocast(enabled=True, device="cuda"):
        if device == "mps":
            return contextlib.nullcontext()
        return _orig_ultralytics_autocast(enabled=enabled, device=device)
    ultralytics.utils.torch_utils.autocast = safe_ultralytics_autocast

    # Monkey-patch TaskAlignedAssigner to run TAL metrics on CPU when using MPS to prevent PyTorch MPS boolean indexing bugs
    import ultralytics.utils.tal
    _orig_tal_forward = ultralytics.utils.tal.TaskAlignedAssigner._forward

    def safe_tal_forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        if pd_scores.device.type == "mps":
            target_device = pd_scores.device
            res = _orig_tal_forward(
                self,
                pd_scores.cpu(),
                pd_bboxes.cpu(),
                anc_points.cpu(),
                gt_labels.cpu(),
                gt_bboxes.cpu(),
                mask_gt.cpu()
            )
            return tuple(x.to(target_device) if isinstance(x, torch.Tensor) else x for x in res)
        return _orig_tal_forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt)

    ultralytics.utils.tal.TaskAlignedAssigner._forward = safe_tal_forward
except Exception:
    pass



def get_default_device():
    """Detects available hardware acceleration (MPS for Apple Silicon, CUDA, or CPU)."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def train_yolo(data_config, model_name, epochs, batch_size, imgsz, device, lr0, project, name, amp=None):
    from ultralytics import YOLO

    if amp is None:
        # Disable AMP on Apple Silicon MPS because torch.amp.autocast does not support 'mps' device_type in PyTorch 2.2.x
        amp = False if str(device).lower() == "mps" else True

    print("🚀 Starting Apple Detection YOLO Training...")
    print(f"📊 Dataset Config: {data_config}")
    print(f"🧠 Initial Weights: {model_name}")
    print(f"⚙️  Epochs: {epochs} | Batch Size: {batch_size} | Image Size: {imgsz}")
    print(f"💻 Hardware Device: {device.upper()} | AMP (Mixed Precision): {amp}")

    # Load pretrained or custom weights
    model = YOLO(model_name)

    # Resolve project path to absolute path to prevent Ultralytics nested path mismatch
    abs_project = str(Path(project).resolve())

    # Train model
    results = model.train(
        data=data_config,
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        device=device,
        lr0=lr0,
        project=abs_project,
        name=name,
        amp=amp,
        exist_ok=True,
        plots=True,
        verbose=True
    )

    # Save best model dynamically to weights/<clean_name>_best.pt
    if hasattr(model, "trainer") and hasattr(model.trainer, "save_dir"):
        save_dir = Path(model.trainer.save_dir)
    else:
        save_dir = Path(project) / name

    best_weights = save_dir / "weights" / "best.pt"
    
    weights_dir = Path("./weights")
    weights_dir.mkdir(parents=True, exist_ok=True)
    
    # Format the destination weights file name dynamically (e.g., orange_model -> orange_best.pt)
    clean_name = name.replace("_model", "")
    destination_weights = weights_dir / f"{clean_name}_best.pt"

    if best_weights.exists():
        shutil.copy2(best_weights, destination_weights)
        print(f"\n🏆 Best model successfully saved to: {destination_weights.resolve()}")
    else:
        print(f"\n⚠️ Trained weights saved in: {save_dir / 'weights'}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Train YOLO model for Apple Detection")
    parser.add_argument("--data", type=str, default="./dataset/apple_dataset/apple_dataset.yaml",
                        help="Path to dataset yaml configuration file")
    parser.add_argument("--model", type=str, default="yolo11n.pt",
                        help="Initial model weights (e.g. yolo11n.pt, yolov8n.pt, or path to .pt file)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--device", type=str, default=get_default_device(),
                        help="Hardware device ('mps', 'cuda', 'cpu', or device ID)")
    parser.add_argument("--amp", type=str, default=None,
                        help="Automatic Mixed Precision ('true' or 'false'). Defaults to false on MPS device.")
    parser.add_argument("--lr0", type=float, default=0.01, help="Initial learning rate")
    parser.add_argument("--project", type=str, default="runs", help="Directory for training output runs")
    parser.add_argument("--name", type=str, default="apple_model", help="Experiment name")

    args = parser.parse_args()

    # Verify dataset yaml exists
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"❌ Error: Dataset config '{args.data}' not found!")
        print("Please run `python3 convert_coco_to_yolo.py` first to prepare the dataset.")
        sys.exit(1)

    amp_setting = None
    if args.amp is not None:
        amp_setting = args.amp.lower() in ("true", "1", "yes")

    train_yolo(
        data_config=str(data_path),
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        lr0=args.lr0,
        project=args.project,
        name=args.name,
        amp=amp_setting
    )


if __name__ == "__main__":
    main()
