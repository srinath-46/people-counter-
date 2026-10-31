"""
Person Detection Module
=======================
Wraps YOLO object detection inference targeting human instances (COCO Class 0).
Provides confidence filtering and non-maximum suppression (NMS) handling.
"""

import sys
from typing import Optional
import numpy as np


class PersonDetector:
    """
    Inference interface for human detection using deep neural networks.

    Attributes:
        model: Ultralytics YOLO model instance.
        conf_thresh: Minimum confidence score required to accept a detection.
        iou_thresh: IoU overlap threshold for non-maximum suppression.
    """

    def __init__(
        self,
        model_path: str = "models/yolov8n.pt",
        conf_thresh: float = 0.35,
        iou_thresh: float = 0.50,
    ) -> None:
        """
        Initializes the detector with model weights and inference parameters.

        Args:
            model_path: Filesystem path to YOLO weights or model identifier.
            conf_thresh: Confidence threshold for person detections.
            iou_thresh: Intersection-over-Union threshold for NMS.
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            sys.stderr.write(
                "[ERROR] Ultralytics framework is not installed.\n"
                "Install dependencies via: pip install ultralytics opencv-python\n"
            )
            sys.exit(1)

        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.model = YOLO(model_path)

    def detect(self, frame: np.ndarray):
        """
        Executes forward inference on a single image frame.

        Args:
            frame: Input video frame as a BGR numpy array.

        Returns:
            Ultralytics Results object containing detected bounding boxes and metadata.
        """
        results = self.model.predict(
            source=frame,
            classes=[0],  # COCO class index for person
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            verbose=False,
        )
        return results[0] if len(results) > 0 else None
