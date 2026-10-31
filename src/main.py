#!/usr/bin/env python3
"""
Retail Footfall Analysis Service - Main Execution Engine
========================================================
Orchestrates detection, tracking, geometric tripwire evaluation,
and event logging across CCTV video inputs.
"""

import argparse
import os
import sys
from typing import Tuple
import cv2
import numpy as np

# Ensure local modules are importable
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tracker import PersonTracker
from counter import DirectionalCounter
from event_logger import EventLogger


def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line options for the tracking service.

    Returns:
        argparse.Namespace populated with configuration values.
    """
    parser = argparse.ArgumentParser(
        description="Retail CCTV People Counter & Live Headcount Service"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="input/store_video.mp4",
        help="Path to source video file or RTSP stream (default: input/store_video.mp4)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="output/annotated_video.mp4",
        help="Destination path for annotated output video (default: output/annotated_video.mp4)",
    )
    parser.add_argument(
        "--csv",
        "-c",
        type=str,
        default="output/events.csv",
        help="Destination path for CSV event log (default: output/events.csv)",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="yolov8n.pt",
        help="Detection model weights (default: yolov8n.pt)",
    )
    parser.add_argument(
        "--tracker",
        type=str,
        default="bytetrack_retail.yaml",
        choices=["bytetrack_retail.yaml", "bytetrack.yaml", "botsort.yaml"],
        help="Tracking algorithm configuration profile (default: bytetrack_retail.yaml)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.28,
        help="Minimum detection confidence threshold (default: 0.28)",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.50,
        help="NMS IoU threshold (default: 0.50)",
    )
    parser.add_argument(
        "--initial-headcount",
        type=int,
        default=8,
        help="Baseline occupants present at video onset (default: 8)",
    )
    parser.add_argument(
        "--line-coords",
        type=str,
        default="0.62,0.77,0.98,0.74",
        help="Normalized tripwire endpoints x1,y1,x2,y2 across entrance gate (default: 0.62,0.77,0.98,0.74)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Render live graphical preview window during execution",
    )
    return parser.parse_args()


def parse_line_coordinates(coords_str: str, frame_width: int, frame_height: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """
    Parses normalized coordinate string into pixel-level line endpoints.

    Args:
        coords_str: Comma-separated normalized floats 'x1,y1,x2,y2'.
        frame_width: Video frame width in pixels.
        frame_height: Video frame height in pixels.

    Returns:
        Tuple of (P1, P2) pixel coordinates.
    """
    try:
        nx1, ny1, nx2, ny2 = map(float, coords_str.split(","))
        p1 = (int(nx1 * frame_width), int(ny1 * frame_height))
        p2 = (int(nx2 * frame_width), int(ny2 * frame_height))
        return p1, p2
    except ValueError as exc:
        raise ValueError(f"Invalid coordinate format '{coords_str}'. Expected 'x1,y1,x2,y2'.") from exc


def main() -> None:
    """Service entrypoint managing the video processing pipeline."""
    args = parse_arguments()

    if not os.path.exists(args.input) and not args.input.startswith(("http://", "https://", "rtsp://")):
        sys.stderr.write(f"[ERROR] Source video '{args.input}' was not found.\n")
        sys.stderr.write("Place footage in input/store_video.mp4 or specify via --input.\n")
        sys.exit(1)

    try:
        from ultralytics import YOLO
    except ImportError:
        sys.stderr.write("[ERROR] Ultralytics is required. Run: pip install ultralytics\n")
        sys.exit(1)

    model_path = args.model
    if not os.path.exists(model_path):
        candidate = os.path.join("models", model_path)
        if os.path.exists(candidate):
            model_path = candidate

    print(f"[Engine] Initializing model '{model_path}'...")
    yolo_model = YOLO(model_path)

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        sys.stderr.write(f"[ERROR] Unable to open video source '{args.input}'.\n")
        sys.exit(1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"[Engine] Input Stream: {width}x{height} @ {fps:.2f} FPS | Total Frames: {total_frames}")

    # Compute tripwire pixel coordinates
    p1, p2 = parse_line_coordinates(args.line_coords, width, height)

    # Initialize tracking, counting, and logging components
    tracker = PersonTracker(
        model=yolo_model,
        tracker_config=args.tracker,
        smooth_alpha=0.35,       # EMA weight for foot-point smoothing
        min_age_frames=4,        # Track must be 4 frames old before crossing fires
    )
    counter = DirectionalCounter(
        line_p1=p1,
        line_p2=p2,
        initial_headcount=args.initial_headcount,
        hysteresis_px=18,
        debounce_frames=int(fps * 1.5),
        confirm_frames=3,        # 3 consecutive frames required to confirm a crossing
    )
    logger = EventLogger(csv_path=args.csv)

    # Configure Video Writer
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    frame_idx = 0
    print("[Engine] Processing video stream...")

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            timestamp_str = logger.format_timestamp(frame_idx, fps)

            # 1. Multi-object tracking
            tracked_objects = tracker.track(frame, conf_thresh=args.conf, iou_thresh=args.iou)

            # 2. Draw Tripwire Line
            counter.draw_tripwire(frame)

            # 3. Process track updates and check line crossings
            detected_in_store = 0
            for obj in tracked_objects:
                track_id   = obj["id"]
                foot_point = obj["foot_point"]     # raw bbox anchor
                smooth_foot = obj["smooth_foot"]   # EMA-smoothed (used for counting)
                bbox       = obj["bbox"]
                conf       = obj["conf"]
                trail      = obj["trail"]           # smoothed trail history
                mature     = obj["mature"]          # age-gate flag

                # Side classification uses the smoothed foot to avoid jitter
                dist = counter.signed_distance(smooth_foot)
                is_inside = (dist < 0)
                if is_inside:
                    detected_in_store += 1
                    box_color = (0, 220, 110)   # Green: Inside Store
                else:
                    box_color = (60, 160, 255)  # Orange: Outside / Entrance

                # Pass smoothed foot + trail + maturity flag into counter
                crossing_event = counter.update(
                    track_id, smooth_foot, frame_idx,
                    trail=trail, mature=mature
                )
                if crossing_event:
                    counter.live_headcount = max(counter.live_headcount, detected_in_store)
                    logger.log_event(timestamp_str, track_id, crossing_event, counter.live_headcount)

                # Render bounding box and identification badge
                x1, y1, x2, y2 = bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                label = f"ID: {track_id} ({conf:.2f})"
                (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(frame, (x1, y1 - lh - 6), (x1 + lw + 6, y1), box_color, -1)
                cv2.putText(frame, label, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1, cv2.LINE_AA)

                # Render smoothed foot dot and trajectory trail
                cv2.circle(frame, smooth_foot, 5, (0, 0, 255), -1)
                cv2.circle(frame, smooth_foot, 5, (255, 255, 255), 1)  # white ring for clarity
                if len(trail) > 1:
                    pts = np.array(trail, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(frame, [pts], False, box_color, 2)

            # Dynamic Headcount Floor:
            # When previously occluded individuals or people out of camera view emerge into visibility,
            # dynamically elevate the active live headcount so it always reflects all visible occupants
            if detected_in_store > counter.live_headcount:
                counter.live_headcount = detected_in_store

            # 4. Render Telemetry HUD
            counter.draw_hud(frame, timestamp_str, fps, detected_in_store=detected_in_store)

            # 5. Write Frame
            out_writer.write(frame)

            if args.show:
                cv2.imshow("Store Occupancy Monitor", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("[Engine] Processing terminated by user.")
                    break

            if frame_idx % 150 == 0:
                progress_pct = (frame_idx / total_frames * 100) if total_frames > 0 else 0
                print(
                    f"  Frame {frame_idx}/{total_frames} ({progress_pct:.1f}%) | "
                    f"IN: {counter.total_in} | OUT: {counter.total_out} | Headcount: {counter.live_headcount} | Detected: {detected_in_store}"
                )

    finally:
        cap.release()
        out_writer.release()
        logger.close()
        if args.show:
            cv2.destroyAllWindows()

    print("\n" + "=" * 55)
    print("Execution Summary:")
    print(f"Total Ingress (IN):   {counter.total_in}")
    print(f"Total Egress  (OUT):  {counter.total_out}")
    print(f"Final Live Headcount: {counter.live_headcount}")
    print(f"Annotated Video:      {args.output}")
    print(f"Audit Log CSV:        {args.csv}")
    print("=" * 55)


if __name__ == "__main__":
    main()
