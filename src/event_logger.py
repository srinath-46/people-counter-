"""
Event Logging Module
====================
Provides streaming persistence of crossing events to delimited CSV format.
Ensures conformance to required audit schema:
timestamp, person_id, event (IN/OUT), headcount
"""

import csv
import os
import sys
from typing import TextIO


class EventLogger:
    """
    Streams pedestrian crossing events to a CSV audit log.

    Attributes:
        csv_path: Target filesystem path for the output CSV.
        file: Open text file handle.
        writer: CSV writer instance.
    """

    def __init__(self, csv_path: str = "output/events.csv") -> None:
        """
        Initializes the CSV output stream and writes the column header.

        Args:
            csv_path: Destination path for the CSV event log.
        """
        self.csv_path = csv_path
        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
        self.file: TextIO = open(csv_path, mode="w", newline="", encoding="utf-8")
        self.writer = csv.writer(self.file)

        # Standard submission schema
        self.writer.writerow(["timestamp", "person_id", "event (IN/OUT)", "headcount"])
        self.file.flush()

    @staticmethod
    def format_timestamp(frame_idx: int, fps: float) -> str:
        """
        Converts frame index and frame rate into a formatted timecode (HH:MM:SS.mmm).

        Args:
            frame_idx: Current frame sequence index.
            fps: Frame rate of the video source.

        Returns:
            Formatted time string.
        """
        total_seconds = frame_idx / fps if fps > 0 else 0.0
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = int(total_seconds % 60)
        milliseconds = int((total_seconds - int(total_seconds)) * 1000)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"

    def log_event(self, timestamp_str: str, person_id: int, event_type: str, headcount: int) -> None:
        """
        Records a verified line crossing event.

        Args:
            timestamp_str: Video timestamp of the crossing event.
            person_id: Tracking ID of the person who crossed.
            event_type: Directional transition ('IN' or 'OUT').
            headcount: Current store occupancy following the event.
        """
        self.writer.writerow([timestamp_str, person_id, event_type, headcount])
        self.file.flush()
        print(f"[{timestamp_str}] Event: {event_type:<3} | Person ID: {person_id:<3} | Headcount: {headcount}")

    def close(self) -> None:
        """Flushes buffers and releases file resources."""
        if self.file and not self.file.closed:
            self.file.close()
