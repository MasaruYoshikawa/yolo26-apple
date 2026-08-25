#!/usr/bin/env python3
"""
Real-time Apple Detection & Live Counter (Webcam / Video Stream)

Processes camera feed or video file with live FPS metrics, bounding boxes,
and real-time apple count overlay.
"""

import sys
import time
import argparse
from pathlib import Path


def run_webcam(source, model_path, conf_thresh, iou_thresh, imgsz, device, save_video, output_path, tracker_config=""):
    import cv2
    from ultralytics import YOLO

    # Load model
    print(f"📦 Loading model: {model_path}")
    model = YOLO(model_path)

    # Initialize video capture (numeric string or int for webcam)
    cap_source = int(source) if str(source).isdigit() else source
    cap = cv2.VideoCapture(cap_source)

    if not cap.isOpened():
        print(f"❌ Error: Unable to open video source: {source}")
        sys.exit(1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_input = cap.get(cv2.CAP_PROP_FPS) or 30.0

    print(f"🎥 Video stream initialized: {width}x{height} @ {fps_input:.1f} FPS")
    print("⌨️  Controls: Press 'q' or ESC to Exit | Press 's' to Save Snapshot")

    writer = None
    if save_video:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(out_path), fourcc, fps_input, (width, height))
        print(f"📼 Recording output to: {out_path.resolve()}")

    prev_frame_time = time.time()
    fps = 0.0
    snapshot_counter = 1

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("🏁 End of video stream reached or frame read error.")
                break

            # Calculate FPS
            curr_frame_time = time.time()
            time_diff = curr_frame_time - prev_frame_time
            if time_diff > 0:
                fps = 1.0 / time_diff
            prev_frame_time = curr_frame_time

            # Run YOLO inference or tracking on frame
            if tracker_config:
                results = model.track(
                    source=frame,
                    conf=conf_thresh,
                    iou=iou_thresh,
                    imgsz=imgsz,
                    device=device,
                    tracker=tracker_config,
                    persist=True,
                    verbose=False
                )[0]
            else:
                results = model.predict(
                    source=frame,
                    conf=conf_thresh,
                    iou=iou_thresh,
                    imgsz=imgsz,
                    device=device,
                    verbose=False
                )[0]

            boxes = results.boxes
            apple_count = len(boxes) if boxes is not None else 0

            # Render bounding boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    cls_name = model.names.get(cls_id, "apple")
                    track_id = int(box.id[0]) if (box.id is not None) else None

                    box_color = (46, 204, 113) # Green
                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

                    label = f"ID:{track_id} {cls_name} {conf:.2f}" if track_id is not None else f"{cls_name} {conf:.2f}"
                    (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(frame, (x1, y1 - text_h - 6), (x1 + text_w + 6, y1), box_color, -1)
                    cv2.putText(frame, label, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

            # Top Header Bar Overlay
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (width, 60), (20, 30, 40), -1)
            cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

            # Accent Bar
            cv2.rectangle(frame, (0, 0), (8, 60), (0, 204, 102), -1)

            # Text status
            cv2.putText(frame, f"Apple Count: {apple_count}", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"FPS: {fps:.1f} | Conf: {conf_thresh:.2f}", (20, 52),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 220, 200), 1, cv2.LINE_AA)

            if writer is not None:
                writer.write(frame)

            cv2.imshow("Apple Real-Time Detection", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27: # q or ESC
                print("👋 Exiting real-time detection.")
                break
            elif key == ord('s'):
                snap_dir = Path("runs/snapshots")
                snap_dir.mkdir(parents=True, exist_ok=True)
                snap_path = snap_dir / f"snapshot_{snapshot_counter:03d}.jpg"
                cv2.imwrite(str(snap_path), frame)
                print(f"📸 Snapshot saved: {snap_path.resolve()}")
                snapshot_counter += 1

    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


def main():
    default_model = "weights/best.pt" if Path("weights/best.pt").exists() else "yolo11n.pt"

    parser = argparse.ArgumentParser(description="Real-time Apple Detection Stream")
    parser.add_argument("--source", type=str, default="0",
                        help="Camera device index (e.g., '0') or video file path")
    parser.add_argument("--model", type=str, default=default_model,
                        help="Path to YOLO model weights (.pt file)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference resolution")
    parser.add_argument("--device", type=str, default="", help="Device ('mps', 'cuda', 'cpu')")
    parser.add_argument("--tracker", type=str, default="", help="Optional tracker config ('bytetrack.yaml' or 'botsort.yaml')")
    parser.add_argument("--save-video", action="store_true", help="Record output video stream to disk")
    parser.add_argument("--output-path", type=str, default="runs/detect/webcam_output.mp4",
                        help="Path to save recorded video")

    args = parser.parse_args()

    run_webcam(
        source=args.source,
        model_path=args.model,
        conf_thresh=args.conf,
        iou_thresh=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        save_video=args.save_video,
        output_path=args.output_path,
        tracker_config=args.tracker
    )


if __name__ == "__main__":
    main()
