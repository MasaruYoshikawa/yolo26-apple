#!/usr/bin/env python3
"""
ByteTrack Video Apple Tracker & Unique Counter

Tracks individual apples in video files or camera streams using ByteTrack,
maintaining persistent Track IDs to count total unique apples without double-counting.
"""

import sys
import time
import argparse
from pathlib import Path


def generate_color_from_id(track_id):
    """Generates a distinct deterministic BGR color based on Track ID."""
    np_hash = (track_id * 1234577 + 999983) % 0xFFFFFF
    r = (np_hash & 0xFF0000) >> 16
    g = (np_hash & 0x00FF00) >> 8
    b = (np_hash & 0x0000FF)
    return (b, g, r)


def draw_hud(frame, current_count, total_unique_count, crossed_count, fps, tracker_type, enable_line, line_pos, axis):
    import cv2
    h, w, _ = frame.shape

    # Resolution scaling factor (relative to standard 720p height)
    scale = max(1.0, h / 720.0)

    banner_h = int(120 * scale)
    bar_w = int(12 * scale)

    # Top translucent HUD banner overlay at top-left
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), (15, 25, 35), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    # Accent left border (vibrant green)
    cv2.rectangle(frame, (0, 0), (bar_w, banner_h), (46, 204, 113), -1)

    # Top Left Large Prominent Counter Text
    title_scale = 1.1 * scale
    sub_scale = 0.85 * scale
    tiny_scale = 0.6 * scale
    thick_main = max(2, int(3.0 * scale))
    thick_sub = max(1, int(2.0 * scale))

    pad_x = int(25 * scale)
    y1 = int(42 * scale)
    y2 = int(80 * scale)
    y3 = int(108 * scale)

    # Top Left Title Banner
    cv2.putText(frame, f"TOTAL UNIQUE APPLES: {total_unique_count}", (pad_x, y1),
                cv2.FONT_HERSHEY_SIMPLEX, title_scale, (0, 255, 127), thick_main, cv2.LINE_AA)

    # Status String (Current Frame & Crossed)
    status_str = f"Active Frame: {current_count} apples"
    if enable_line:
        status_str += f"  |  Line Crossed: {crossed_count}"
    
    cv2.putText(frame, status_str, (pad_x, y2),
                cv2.FONT_HERSHEY_SIMPLEX, sub_scale, (255, 255, 255), thick_sub, cv2.LINE_AA)

    sub_str = f"FPS: {fps:.1f}  |  Tracker: {tracker_type.upper()}  |  Resolution: {w}x{h}"
    cv2.putText(frame, sub_str, (pad_x, y3),
                cv2.FONT_HERSHEY_SIMPLEX, tiny_scale, (180, 210, 200), max(1, int(1.5 * scale)), cv2.LINE_AA)

    # Draw counting line if enabled
    if enable_line:
        line_color = (0, 215, 255) # Bright Yellow/Orange
        line_thick = max(2, int(3 * scale))
        if axis == "horizontal":
            line_y = int(h * line_pos)
            cv2.line(frame, (0, line_y), (w, line_y), line_color, line_thick)
            cv2.putText(frame, "COUNTING LINE", (int(15 * scale), line_y - int(12 * scale)),
                        cv2.FONT_HERSHEY_SIMPLEX, sub_scale, line_color, thick_sub, cv2.LINE_AA)
        else:
            line_x = int(w * line_pos)
            cv2.line(frame, (line_x, 0), (line_x, h), line_color, line_thick)
            cv2.putText(frame, "COUNTING LINE", (line_x + int(12 * scale), int(140 * scale)),
                        cv2.FONT_HERSHEY_SIMPLEX, sub_scale, line_color, thick_sub, cv2.LINE_AA)

    return frame


def resolve_source(source_str):
    """Resolves webcam index, local file path, or YouTube video URL using yt-dlp."""
    if str(source_str).startswith(("http://", "https://", "www.youtube.com", "youtu.be")):
        print(f"🌐 YouTube video URL detected: {source_str}")
        try:
            import yt_dlp
            ydl_opts = {
                'format': 'best[ext=mp4]/best',
                'quiet': True,
                'no_warnings': True
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(source_str, download=False)
                stream_url = info.get('url')
                title = info.get('title', 'YouTube Stream')
                print(f"🎬 YouTube Video Title: '{title}'")
                if stream_url:
                    return stream_url
        except Exception as e:
            print(f"⚠️ Note: Direct stream extraction via yt_dlp error: {e}")
            print("💡 Tip: You can download the video first using `yt-dlp <URL>` and pass the downloaded .mp4 file.")
    
    return int(source_str) if str(source_str).isdigit() else source_str


def run_tracker(source, model_path, tracker_config, conf_thresh, iou_thresh, imgsz, device,
                enable_line, line_pos, line_axis, save_video, output_path, show, max_seconds=0.0, max_frames=0):
    import cv2
    from ultralytics import YOLO

    source_path = str(source)
    cap_source = resolve_source(source_path)

    cap = cv2.VideoCapture(cap_source)
    if not cap.isOpened():
        print(f"❌ Error: Unable to open video source '{source}'")
        sys.exit(1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_input = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # Determine frame limit if max_seconds specified
    if max_seconds > 0:
        calculated_limit = int(fps_input * max_seconds)
        max_frames = calculated_limit if max_frames <= 0 else min(max_frames, calculated_limit)

    print("=" * 60)
    print("🚀 STARTING BYTETRACK VIDEO APPLE TRACKING")
    print("=" * 60)
    print(f"🎥 Video Input   : {source}")
    print(f"📐 Resolution    : {width}x{height} @ {fps_input:.1f} FPS")
    print(f"🎯 Target Limit  : {f'{max_frames} frames (~{max_seconds if max_seconds > 0 else max_frames/fps_input:.1f}s)' if max_frames > 0 else 'Full Video'}")
    print(f"🧠 Model Weights : {model_path}")
    print(f"⚙️  Confidence   : {conf_thresh:.2f}")
    print("=" * 60)

    model = YOLO(model_path)

    writer = None
    if save_video:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(out_p), fourcc, fps_input, (width, height))
        print(f"📼 Saving tracked output video to: {out_p.resolve()}")

    unique_track_ids = set()
    track_positions = {}  # {track_id: previous_center_coord}
    crossed_track_ids = set()

    prev_frame_time = time.time()
    fps = 0.0
    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            if max_frames > 0 and frame_idx > max_frames:
                print(f"⏱️ Reached frame limit of {max_frames} frames ({max_seconds if max_seconds > 0 else max_frames/fps_input:.1f}s). Stopping.")
                break
            curr_time = time.time()
            dt = curr_time - prev_frame_time
            if dt > 0:
                fps = 1.0 / dt
            prev_frame_time = curr_time

            # Run YOLO with ByteTrack persistent multi-object tracking
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

            boxes = results.boxes
            current_frame_count = 0

            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    cls_name = model.names.get(cls_id, "apple")

                    # Track ID assignment from ByteTrack
                    track_id = int(box.id[0]) if box.id is not None else None

                    if track_id is not None:
                        unique_track_ids.add(track_id)
                        box_color = generate_color_from_id(track_id)
                        label = f"ID:{track_id} {cls_name} {conf:.2f}"

                        # Line crossing logic
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2
                        curr_pos = center_y if line_axis == "horizontal" else center_x
                        line_coord = int((height if line_axis == "horizontal" else width) * line_pos)

                        if track_id in track_positions:
                            prev_pos = track_positions[track_id]
                            # Check if object crossed line boundary in either direction
                            if (prev_pos < line_coord <= curr_pos) or (curr_pos <= line_coord < prev_pos):
                                crossed_track_ids.add(track_id)
                        track_positions[track_id] = curr_pos

                    else:
                        box_color = (46, 204, 113)
                        label = f"{cls_name} {conf:.2f}"

                    current_frame_count += 1

                    # Draw Bounding Box (dynamically scaled for 4K / HD resolutions)
                    scale_res = max(1.0, height / 720.0)
                    box_thick = max(2, int(2.5 * scale_res))
                    font_scale = 0.55 * scale_res
                    font_thick = max(1, int(1.5 * scale_res))
                    pad_box = int(6 * scale_res)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, box_thick)

                    # Label badge
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thick)
                    cv2.rectangle(frame, (x1, y1 - th - pad_box), (x1 + tw + pad_box, y1), box_color, -1)
                    cv2.putText(frame, label, (x1 + int(3 * scale_res), y1 - int(3 * scale_res)),
                                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thick, cv2.LINE_AA)

            # Render HUD
            frame = draw_hud(
                frame=frame,
                current_count=current_frame_count,
                total_unique_count=len(unique_track_ids),
                crossed_count=len(crossed_track_ids),
                fps=fps,
                tracker_type=tracker_config.replace('.yaml', ''),
                enable_line=enable_line,
                line_pos=line_pos,
                axis=line_axis
            )

            if writer is not None:
                writer.write(frame)

            if show:
                cv2.imshow("ByteTrack Apple Video Tracking", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    print("👋 Stopping tracking stream.")
                    break

            if frame_idx % 30 == 0:
                print(f" Frame [{frame_idx}/{total_frames if total_frames > 0 else 'live'}] | Current: {current_frame_count} | Unique Total: {len(unique_track_ids)}")

    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if show:
            cv2.destroyAllWindows()

    print("\n" + "=" * 60)
    print("📊 BYTETRACK TRACKING FINAL SUMMARY")
    print("=" * 60)
    print(f"Total Video Frames Processed : {frame_idx}")
    print(f"Total Unique Apples Tracked  : {len(unique_track_ids)}")
    if enable_line:
        print(f"Total Apples Line Crossed    : {len(crossed_track_ids)}")
    if save_video and writer is not None:
        print(f"Recorded Video Saved To      : {Path(output_path).resolve()}")
    print("=" * 60)


def main():
    default_model = "weights/best.pt" if Path("weights/best.pt").exists() else "yolo11n.pt"

    parser = argparse.ArgumentParser(description="ByteTrack Video Apple Tracker & Counter")
    parser.add_argument("--source", type=str, required=True,
                        help="Path to video file (e.g. video.mp4) or camera index '0'")
    parser.add_argument("--model", type=str, default=default_model,
                        help="Path to model weights file (.pt)")
    parser.add_argument("--tracker", type=str, default="bytetrack.yaml",
                        help="Tracker config ('bytetrack.yaml' or 'botsort.yaml')")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference resolution")
    parser.add_argument("--device", type=str, default="", help="Device ('mps', 'cuda', 'cpu')")
    
    # Line Counting options
    parser.add_argument("--enable-line", action="store_true", help="Enable ROI counting line")
    parser.add_argument("--line-position", type=float, default=0.5,
                        help="Counting line position ratio (0.0 to 1.0, default 0.5)")
    parser.add_argument("--line-axis", type=str, default="horizontal", choices=["horizontal", "vertical"],
                        help="Counting line orientation ('horizontal' or 'vertical')")
    
    # Output & Display options
    parser.add_argument("--max-seconds", type=float, default=0.0,
                        help="Limit processing to first N seconds of video (e.g. --max-seconds 10)")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="Limit processing to first N frames of video (e.g. --max-frames 300)")
    parser.add_argument("--save-video", action="store_true", help="Save output video with tracking visualization")
    parser.add_argument("--output-path", type=str, default="runs/track/apple_tracked.mp4",
                        help="Output path for recorded tracked video")
    parser.add_argument("--show", action="store_true", help="Display tracking GUI window during processing")

    args = parser.parse_args()

    run_tracker(
        source=args.source,
        model_path=args.model,
        tracker_config=args.tracker,
        conf_thresh=args.conf,
        iou_thresh=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        enable_line=args.enable_line,
        line_pos=args.line_position,
        line_axis=args.line_axis,
        save_video=args.save_video,
        output_path=args.output_path,
        show=args.show,
        max_seconds=args.max_seconds,
        max_frames=args.max_frames
    )


if __name__ == "__main__":
    main()
