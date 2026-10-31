"""
Multi-Object Tracking Module
============================
Manages continuous identity association and trajectory tracking for detected persons.

Enhancements over base implementation:
  - Foot-point EMA smoothing: reduces jitter on the counting line from detector noise.
  - Track age gating: suppresses crossing events for freshly-born ghost tracks.
  - Stable trail history for trajectory visualisation.
"""

from collections import defaultdict, deque
from typing import Any, Dict, List, Tuple
import numpy as np


class PersonTracker:
    """
    Tracks detected human instances across sequential video frames.

    Attributes:
        model: Underlying detection/tracking model instance.
        tracker_config: Tracking configuration profile (e.g., 'bytetrack_retail.yaml').
        max_trail_len: Maximum historical positions retained for each track.
        smooth_alpha: EMA smoothing factor for foot-point coordinates (0 = no update, 1 = no smoothing).
        min_age_frames: Minimum track age (frames) before crossing events are evaluated.
        track_history: Mapping of track ID → deque of raw foot positions.
        track_smoothed: Mapping of track ID → EMA-smoothed foot position.
        track_age: Mapping of track ID → number of frames seen.
    """

    def __init__(
        self,
        model: Any,
        tracker_config: str = "bytetrack_retail.yaml",
        max_trail_len: int = 45,
        smooth_alpha: float = 0.35,
        min_age_frames: int = 4,
    ) -> None:
        """
        Initialises the tracking manager with smoothing and age-gating.

        Args:
            model: Instantiated YOLO model capable of tracking.
            tracker_config: Tracker profile path or name.
            max_trail_len: Maximum frames of trajectory history per track.
            smooth_alpha: EMA weight for the newest foot point (0 < alpha ≤ 1).
                          Lower = heavier smoothing, higher = more responsive.
            min_age_frames: Frames a track must exist before it can trigger a crossing event.
        """
        self.model = model
        self.tracker_config = tracker_config
        self.max_trail_len = max_trail_len
        self.smooth_alpha = smooth_alpha
        self.min_age_frames = min_age_frames

        self.track_history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=self.max_trail_len))
        self.track_smoothed: Dict[int, Tuple[float, float]] = {}
        self.track_age: Dict[int, int] = defaultdict(int)

    def _smooth_foot(self, track_id: int, raw_foot: Tuple[int, int]) -> Tuple[int, int]:
        """
        Applies Exponential Moving Average to the raw foot-point coordinate.
        Suppresses per-frame bbox jitter that can cause false line crossings.

        Args:
            track_id: Track identifier.
            raw_foot: Raw (x, y) foot point from detector.

        Returns:
            Smoothed (x, y) foot point as integers.
        """
        rx, ry = float(raw_foot[0]), float(raw_foot[1])
        if track_id not in self.track_smoothed:
            self.track_smoothed[track_id] = (rx, ry)
        else:
            sx, sy = self.track_smoothed[track_id]
            sx = self.smooth_alpha * rx + (1.0 - self.smooth_alpha) * sx
            sy = self.smooth_alpha * ry + (1.0 - self.smooth_alpha) * sy
            self.track_smoothed[track_id] = (sx, sy)
        sx, sy = self.track_smoothed[track_id]
        return (int(sx), int(sy))

    def track(
        self,
        frame: np.ndarray,
        conf_thresh: float = 0.28,
        iou_thresh: float = 0.45,
    ) -> List[Dict[str, Any]]:
        """
        Performs detection + ByteTrack association on a single frame.

        Args:
            frame: Current video frame (BGR numpy array).
            conf_thresh: Minimum detection confidence score.
            iou_thresh: NMS IoU suppression threshold.

        Returns:
            List of track dicts:
              - 'id'           : int   – persistent track identifier
              - 'age'          : int   – frames this track has been alive
              - 'bbox'         : (x1, y1, x2, y2) pixel bounding box
              - 'conf'         : float – detection confidence
              - 'foot_point'   : (x, y) raw bottom-centre anchor
              - 'smooth_foot'  : (x, y) EMA-smoothed foot point (use for line crossing)
              - 'trail'        : list[(x, y)] recent smoothed foot positions
              - 'mature'       : bool  – True if track is old enough to fire events
        """
        results = self.model.track(
            source=frame,
            persist=True,
            classes=[0],
            conf=conf_thresh,
            iou=iou_thresh,
            tracker=self.tracker_config,
            verbose=False,
        )

        active_tracks: List[Dict[str, Any]] = []

        if (
            len(results) > 0
            and results[0].boxes is not None
            and results[0].boxes.id is not None
        ):
            boxes = results[0].boxes
            track_ids = boxes.id.int().cpu().tolist()
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()

            for box, track_id, conf in zip(xyxy, track_ids, confs):
                x1, y1, x2, y2 = map(int, box)

                # Raw ground-plane anchor: bottom-centre of bounding box
                raw_foot = (int((x1 + x2) / 2), int(y2))

                # EMA-smoothed foot for stable line-crossing geometry
                smooth_foot = self._smooth_foot(track_id, raw_foot)

                # Age tracking
                self.track_age[track_id] += 1
                age = self.track_age[track_id]

                # Append smoothed position to trail for visualisation
                self.track_history[track_id].append(smooth_foot)

                active_tracks.append({
                    "id":          track_id,
                    "age":         age,
                    "bbox":        (x1, y1, x2, y2),
                    "conf":        float(conf),
                    "foot_point":  raw_foot,
                    "smooth_foot": smooth_foot,
                    "trail":       list(self.track_history[track_id]),
                    "mature":      age >= self.min_age_frames,
                })

        return active_tracks
