#!/usr/bin/env python3
"""
Apple Object Detection & Object Counter Script

Runs inference on single images or directories of images using trained YOLO weights.
Renders bounding boxes, confidence scores, class labels, and an Apple Count overlay banner.
"""

import os
import sys
import argparse
from pathlib import Path


def get_fruit_color(fruit_name):
    """Returns distinct BGR colors for different fruit targets."""
    name = str(fruit_name).lower()
    if "orange" in name:
        return (30, 144, 255)    # Vibrant Amber / Orange (BGR)
    elif "blueberry" in name or "berry" in name:
        return (225, 105, 65)    # Royal Indigo / Violet (BGR)
    elif "apple" in name:
        return (113, 204, 46)    # Emerald Green (BGR)
    else:
        return (113, 204, 46)    # Default Theme Accent


def draw_fruit_count_banner(image, count, conf_threshold, fruit_name="Fruit"):
    """Draws a clean top banner displaying the total number of detected fruits."""
    import cv2
    h, w, _ = image.shape
    banner_height = 60
    
    # Create translucent dark overlay for banner header
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_height), (20, 30, 40), -1)
    cv2.addWeighted(overlay, 0.85, image, 0.15, 0, image)

    # Accent left border bar based on fruit type
    accent_color = get_fruit_color(fruit_name)
    cv2.rectangle(image, (0, 0), (8, banner_height), accent_color, -1)

    # Capitalize fruit name
    display_name = fruit_name.capitalize() if fruit_name else "Fruit"

    # Banner text
    title_text = f"{display_name} Detection | Count: {count}"
    sub_text = f"Conf Thresh: {conf_threshold:.2f}"

    cv2.putText(image, title_text, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image, sub_text, (20, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 220, 200), 1, cv2.LINE_AA)

    return image


def run_detection(source, model_path, conf_thresh, iou_thresh, imgsz, output_dir, device, show, fruit_name=None, progress_callback=None, log_callback=None):
    import cv2
    from ultralytics import YOLO

    def log(msg):
        if log_callback:
            log_callback(msg)
        else:
            print(msg)

    source_path = Path(source)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    log(f"📦 Loading model from: {model_path}")
    model = YOLO(model_path)

    # Determine primary fruit name if not specified
    if not fruit_name:
        if hasattr(model, 'names') and model.names:
            first_cls = str(model.names.get(0, "Fruit")).strip()
            fruit_name = first_cls if first_cls and first_cls.lower() != "0" else "Fruit"
        else:
            fruit_name = "Fruit"

    log(f"🎯 Target Object: {fruit_name.capitalize()}")

    # Collect target images
    if source_path.is_file():
        image_paths = [source_path]
    elif source_path.is_dir():
        valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}
        image_paths = [p for p in source_path.rglob('*') if p.suffix.lower() in valid_exts]
    else:
        log(f"❌ Error: Source path '{source}' does not exist.")
        sys.exit(1)

    if not image_paths:
        log(f"⚠️ No image files found in: {source_path}")
        return

    log(f"📷 Processing {len(image_paths)} images...")
    total_count = 0
    results_list = []
    box_color = get_fruit_color(fruit_name)

    for idx, img_path in enumerate(image_paths, 1):
        # Run YOLO inference
        results = model.predict(
            source=str(img_path),
            conf=conf_thresh,
            iou=iou_thresh,
            imgsz=imgsz,
            device=device,
            verbose=False
        )[0]

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        boxes = results.boxes
        det_count = len(boxes) if boxes is not None else 0
        total_count += det_count
        results_list.append({
            'image_name': img_path.name,
            'fruit_count': det_count
        })

        # Render custom bounding boxes
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = model.names.get(cls_id, fruit_name)

                cv2.rectangle(img, (x1, y1), (x2, y2), box_color, 2)

                # Label text
                label = f"{cls_name} {conf:.2f}"
                (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

                # Label background rectangle
                cv2.rectangle(img, (x1, y1 - text_h - 6), (x1 + text_w + 6, y1), box_color, -1)
                cv2.putText(img, label, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        # Render Count Header Banner
        img = draw_fruit_count_banner(img, det_count, conf_thresh, fruit_name=fruit_name)

        # Save annotated image
        save_path = out_dir / f"pred_{img_path.name}"
        cv2.imwrite(str(save_path), img)

        log(f"  [{idx}/{len(image_paths)}] {img_path.name}: {det_count} {fruit_name}(s) detected -> Saved: {save_path.name}")
        if progress_callback:
            progress_callback(idx, len(image_paths))

        if show:
            cv2.imshow(f"{fruit_name.capitalize()} Detection", img)
            if cv2.waitKey(0) & 0xFF == ord('q'):
                break

    if show:
        cv2.destroyAllWindows()

    avg_count = total_count / len(image_paths) if image_paths else 0

    # Write results to CSV
    csv_path = out_dir / "カウント結果.csv"
    import csv
    try:
        with open(csv_path, mode='w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(['画像ファイル名', '検出個数'])
            for res in results_list:
                writer.writerow([res['image_name'], res['fruit_count']])
            writer.writerow([])
            writer.writerow(['対象画像総数', len(image_paths)])
            writer.writerow(['検出総数', total_count])
            writer.writerow(['平均検出数（画像あたり）', f"{avg_count:.2f}"])
        log(f"📊 CSV results saved to: {csv_path.resolve()}")
    except Exception as e:
        log(f"⚠️ Warning: Failed to write CSV results: {e}")

    log("\n" + "=" * 50)
    log("📊 DETECTION SUMMARY")
    log("=" * 50)
    log(f"Total Images Processed : {len(image_paths)}")
    log(f"Total {fruit_name.capitalize()}s Detected  : {total_count}")
    log(f"Average {fruit_name.capitalize()}s / Image : {avg_count:.2f}")
    log(f"Annotated Outputs Saved: {out_dir.resolve()}")
    log("=" * 50)


def main():
    # Find default model path if available
    default_model = "weights/best.pt" if Path("weights/best.pt").exists() else "yolo11n.pt"

    parser = argparse.ArgumentParser(description="Run Apple Detection on images")
    parser.add_argument("--source", type=str, default="./dataset/apple_dataset/images/test",
                        help="Path to image file or directory of images")
    parser.add_argument("--model", type=str, default=default_model,
                        help="Path to model weights (.pt file)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--output-dir", type=str, default="runs/detect/predictions",
                        help="Directory to save output images")
    parser.add_argument("--device", type=str, default="", help="Device ('mps', 'cuda', 'cpu')")
    parser.add_argument("--show", action="store_true", help="Display images in GUI window")

    args = parser.parse_args()

    run_detection(
        source=args.source,
        model_path=args.model,
        conf_thresh=args.conf,
        iou_thresh=args.iou,
        imgsz=args.imgsz,
        output_dir=args.output_dir,
        device=args.device,
        show=args.show
    )


if __name__ == "__main__":
    main()
