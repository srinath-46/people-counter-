# Retail Store Occupancy & Bi-Directional Footfall Tracking System

A modular computer vision service designed for retail store CCTV feeds to monitor live occupancy, track bidirectional customer traffic (ingress/egress), and record timestamped crossing events.

The architecture integrates YOLOv8 for human detection with ByteTrack for persistent identity association across visual occlusions, paired with a geometric vector-crossing state machine.

---

## Directory Structure

```text
Store-People-Counter/
├── models/
│   └── yolov8n.pt               # YOLO object detection weights
├── input/
│   └── store_video.mp4          # Source video footage
├── output/
│   ├── annotated_video.mp4      # Rendered video with bounding boxes, IDs, and HUD
│   └── events.csv               # Event audit log (timestamp, person_id, event, headcount)
├── src/
│   ├── detector.py              # Object detection wrapper
│   ├── tracker.py               # Multi-object tracking and trajectory management
│   ├── counter.py               # Vector tripwire and state machine
│   ├── event_logger.py          # CSV event streaming and formatting
│   └── main.py                  # Service entrypoint and CLI orchestration
├── README.md                    # System documentation
├── requirements.txt             # Python runtime dependencies
└── approach_note.pdf            # Engineering report on methodology and edge cases
```

---

## Environment Setup

### Prerequisites
- Python 3.9+
- FFmpeg (for video encoding)
- Optional: CUDA-enabled GPU for accelerated inference

### Installation
```bash
git clone <repository-url>
cd Store-People-Counter

python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

---

## Video Acquisition

To download the reference CCTV retail footage into the `input/` directory:

```bash
yt-dlp -f "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best" "https://youtu.be/KMJS66jBtVQ" -o "input/store_video.mp4"
```

---

## Usage

### Standard Batch Execution
Process the input video, generate the annotated output file, and stream events to CSV:

```bash
python src/main.py --input input/store_video.mp4 --output output/annotated_video.mp4 --csv output/events.csv --initial-headcount 6
```

### Real-Time Display Mode
To inspect inference and tracking in real time with an active OpenCV window:

```bash
python src/main.py --input input/store_video.mp4 --output output/annotated_video.mp4 --csv output/events.csv --show
```

### Command-Line Arguments Reference

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--input`, `-i` | `str` | `input/store_video.mp4` | Path to source video file or RTSP stream URL |
| `--output`, `-o` | `str` | `output/annotated_video.mp4` | Destination path for annotated output video |
| `--csv`, `-c` | `str` | `output/events.csv` | Destination path for event audit log |
| `--model`, `-m` | `str` | `yolov8n.pt` | Detection model weights (yolov8n.pt, yolov8s.pt) |
| `--tracker` | `str` | `bytetrack.yaml` | Multi-object tracking configuration file |
| `--conf` | `float` | `0.35` | Minimum confidence threshold for human detection |
| `--iou` | `float` | `0.50` | Non-Maximum Suppression (NMS) IoU threshold |
| `--initial-headcount` | `int` | `6` | Calibrated customer/staff baseline at t=0 |
| `--line-coords` | `str` | `0.55,0.85,0.98,0.72` | Normalized coordinates (x1,y1,x2,y2) for tripwire |
| `--show` | `flag` | `False` | Enables interactive graphical window |

---

## Output Specifications

### Event Log CSV (`output/events.csv`)
Events are streamed sequentially as directional line crossings are verified:

```csv
timestamp,person_id,event (IN/OUT),headcount
00:00:04.150,7,IN,7
00:00:05.800,8,IN,8
00:00:06.450,9,IN,9
00:00:18.200,10,IN,10
00:00:19.500,11,IN,11
00:00:44.800,1,OUT,10
```

- **`timestamp`**: Media playback timecode formatted as `HH:MM:SS.mmm`.
- **`person_id`**: Persistent integer identifier assigned by the tracker.
- **`event (IN/OUT)`**: Directional classification (`IN` for entry, `OUT` for exit).
- **`headcount`**: Current active occupancy within the premises after event resolution.

### Annotated Video (`output/annotated_video.mp4`)
- Ground-anchored bounding boxes and persistent tracking identifiers.
- Directional counting tripwire across the store entrance threshold.
- Real-time telemetry Heads-Up Display (HUD) showing live headcount, cumulative entries, and cumulative exits.

---

## Edge Case Handling

1. **Visual Occlusion**: ByteTrack utilizes two-stage Hungarian association to match low-confidence detections against Kalman filter state forecasts, preserving track identity during temporary blockages behind merchandise displays.
2. **Concurrent Group Entry**: Floor-contact anchor points (bottom-center of bounding boxes) remain distinct even when upper-body boxes overlap, enabling parallel counting across simultaneous entries.
3. **Threshold Lingering**: A spatial hysteresis band ($\pm 18\text{px}$) combined with a $1.5\text{s}$ debounce cooldown prevents false state flipping when individuals pause at the doorway.
4. **Staff Filtering**: Cashier operations are confined to a designated polygon zone, preventing stationary staff movements from affecting customer traffic metrics.
