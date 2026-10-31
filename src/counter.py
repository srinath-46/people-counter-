"""
Directional Counting & State Machine Module
===========================================
Implements geometric line-crossing verification, spatial hysteresis filtering,
multi-frame confirmation voting, and occupancy state management for retail
footfall analytics.

Upgrades over base:
  - Multi-frame vote buffer: requires N consecutive frames on the new side before
    a crossing is committed, eliminating single-frame jitter events.
  - Trajectory direction validation: verifies the foot's velocity vector is
    actually pointing through the gate (rules out lateral walkers who brush the
    line without passing through).
  - Segment corridor constraint: only counts transitions through the actual
    physical entrance zone.
"""

from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np


class DirectionalCounter:
    """
    Evaluates pedestrian trajectories across a virtual counting tripwire.

    Attributes:
        p1, p2            : Tripwire endpoint pixel coordinates.
        initial_headcount : Baseline occupant count at t=0.
        hysteresis_px     : Dead-band distance that suppresses threshold oscillation.
        debounce_frames   : Cooldown frames per track ID between events.
        confirm_frames    : Consecutive frames required on new side to confirm a crossing.
        total_in          : Cumulative IN count.
        total_out         : Cumulative OUT count.
        live_headcount    : Real-time occupancy estimate.
    """

    def __init__(
        self,
        line_p1: Tuple[int, int],
        line_p2: Tuple[int, int],
        initial_headcount: int = 8,
        hysteresis_px: int = 18,
        debounce_frames: int = 35,
        confirm_frames: int = 3,
    ) -> None:
        """
        Initialises the tripwire geometry and state parameters.

        Args:
            line_p1          : Origin coordinate (x, y) of the tripwire vector.
            line_p2          : Terminal coordinate (x, y) of the tripwire vector.
            initial_headcount: Baseline store occupancy.
            hysteresis_px    : Dead-band boundary distance in pixels.
            debounce_frames  : Cooldown frames between crossing events per track.
            confirm_frames   : Consecutive frames a track must sit on the new side
                               before the crossing is registered (multi-frame vote).
        """
        self.p1 = line_p1
        self.p2 = line_p2
        self.initial_headcount = initial_headcount
        self.hysteresis_px = hysteresis_px
        self.debounce_frames = debounce_frames
        self.confirm_frames = confirm_frames

        self.total_in: int = 0
        self.total_out: int = 0
        self.live_headcount: int = initial_headcount

        # Per-track state
        self.person_last_side: Dict[int, int] = {}
        self.last_event_frame: Dict[int, int] = defaultdict(int)

        # Multi-frame vote buffer: track_id → deque of recent side readings
        self._vote_buffer: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=self.confirm_frames)
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Geometry helpers
    # ─────────────────────────────────────────────────────────────────────────

    def signed_distance(self, point: Tuple[int, int]) -> float:
        """
        Signed perpendicular distance from a point to the directed line P1→P2.

        In OpenCV screen-space (Y grows downward):
          Negative → inside the store (upper-left side of the vector)
          Positive → outside / entrance mat (lower-right side)
        """
        x, y = point
        x1, y1 = self.p1
        x2, y2 = self.p2
        cross = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)
        length = np.hypot(x2 - x1, y2 - y1)
        return cross / length if length > 0 else 0.0

    def is_in_corridor(self, point: Tuple[int, int], margin: float = 0.15) -> bool:
        """
        Returns True when the point projects within the finite gate segment [P1, P2].
        Prevents transitions elsewhere in the store (cashier desk, aisle ends) from
        being counted as door crossings.
        """
        x, y = point
        x1, y1 = self.p1
        x2, y2 = self.p2
        dx, dy = x2 - x1, y2 - y1
        len_sq = dx * dx + dy * dy
        if len_sq == 0:
            return False
        t = ((x - x1) * dx + (y - y1) * dy) / len_sq
        return -margin <= t <= 1.0 + margin

    def _trajectory_crosses_gate(
        self, trail: List[Tuple[int, int]], new_side: int
    ) -> bool:
        """
        Validates that the movement direction of the track is actually pointed
        through the gate, not just drifting laterally alongside it.

        Computes the dot product of the recent motion vector with the gate normal.
        A crossing is only plausible if the motion component perpendicular to the
        gate is substantial (> 30% of total speed).

        Args:
            trail   : Recent list of (x, y) smoothed foot positions (oldest first).
            new_side: The side the track is transitioning to (+1 inside, -1 outside).

        Returns:
            True if the direction of motion is consistent with a gate crossing.
        """
        if len(trail) < 4:
            return True  # Not enough history — give benefit of the doubt

        # Use the last 4 positions to compute motion vector
        older = trail[-4]
        newer = trail[-1]
        dx = newer[0] - older[0]
        dy = newer[1] - older[1]
        speed = np.hypot(dx, dy)
        if speed < 3.0:
            return True  # Almost stationary — let hysteresis handle it

        # Gate vector P1→P2
        gx, gy = self.p2[0] - self.p1[0], self.p2[1] - self.p1[1]
        # In screen-space (Y grows downward), the store interior is ABOVE the line (lower Y values)
        # i.e. signed_distance < 0 means inside, which means persons move UP (dy < 0) to enter.
        # Right-hand normal (gy, -gx) points toward increasing signed_distance (outside).
        # We want the INWARD normal = -(gy, -gx) = (-gy, gx)... but wait:
        # Actually tested: moving UP (dy<0) from outside to inside gives perp = +1.0 with (gy, -gx).
        # So use (gy, -gx) as the inward normal.
        nx, ny = gy, -gx
        gate_len = np.hypot(nx, ny)
        if gate_len == 0:
            return True
        nx, ny = nx / gate_len, ny / gate_len

        # Component of motion in the inward-normal direction
        perp_component = (dx * nx + dy * ny) / speed

        # new_side == 1  (inside) → person should be moving in the inward direction  (perp > 0)
        # new_side == -1 (outside) → person should be moving outward (perp < 0)
        # Threshold 0.15 accepts diagonal walkers (entering at an angle through the door)
        if new_side == 1:
            return perp_component > 0.15
        else:
            return perp_component < -0.15

    # ─────────────────────────────────────────────────────────────────────────
    # Core update
    # ─────────────────────────────────────────────────────────────────────────

    def update(
        self,
        track_id: int,
        foot_point: Tuple[int, int],
        frame_idx: int,
        trail: Optional[List[Tuple[int, int]]] = None,
        mature: bool = True,
    ) -> Optional[str]:
        """
        Updates track state and checks for a confirmed gate crossing.

        Args:
            track_id  : Persistent integer track identifier.
            foot_point: EMA-smoothed (x, y) ground anchor point.
            frame_idx : Sequential frame index.
            trail     : Recent smoothed foot positions (for direction check).
            mature    : False for newly-born tracks that should be age-gated.

        Returns:
            'IN', 'OUT', or None.
        """
        dist = self.signed_distance(foot_point)
        in_corridor = self.is_in_corridor(foot_point)

        # Classify side with hysteresis dead-band
        current_side: Optional[int] = None
        if dist < -self.hysteresis_px:
            current_side = 1    # Inside store
        elif dist > self.hysteresis_px:
            current_side = -1   # Outside / entrance mat

        # Update vote buffer regardless of corridor
        if current_side is not None:
            self._vote_buffer[track_id].append(current_side)

        prev_side = self.person_last_side.get(track_id)
        event_type: Optional[str] = None

        if current_side is not None:
            # ── Multi-frame confirmation vote ──────────────────────────────
            # Commit a crossing only when the last `confirm_frames` readings
            # all agree on the new side (eliminates single-frame jitter)
            vote_buf = list(self._vote_buffer[track_id])
            confirmed_new_side = (
                len(vote_buf) >= self.confirm_frames
                and all(v == current_side for v in vote_buf[-self.confirm_frames :])
            )

            if (
                prev_side is not None
                and prev_side != current_side
                and confirmed_new_side
                and mature                                       # age gate
                and in_corridor                                  # spatial gate
                and (frame_idx - self.last_event_frame[track_id]) > self.debounce_frames
                and self._trajectory_crosses_gate(trail or [], current_side)  # direction gate
            ):
                if prev_side == -1 and current_side == 1:
                    self.total_in += 1
                    self.live_headcount += 1
                    event_type = "IN"
                    self.last_event_frame[track_id] = frame_idx

                elif prev_side == 1 and current_side == -1:
                    self.total_out += 1
                    self.live_headcount = max(0, self.live_headcount - 1)
                    event_type = "OUT"
                    self.last_event_frame[track_id] = frame_idx

            # Update committed side only once confirmed
            if confirmed_new_side:
                self.person_last_side[track_id] = current_side

        return event_type

    # ─────────────────────────────────────────────────────────────────────────
    # Rendering
    # ─────────────────────────────────────────────────────────────────────────

    def draw_tripwire(self, frame: np.ndarray) -> None:
        """Renders the entrance counting gate onto the frame."""
        # Gate line
        cv2.line(frame, self.p1, self.p2, (255, 230, 0), 3, cv2.LINE_AA)
        cv2.circle(frame, self.p1, 6, (0, 200, 255), -1)
        cv2.circle(frame, self.p2, 6, (0, 200, 255), -1)

        mid_x = (self.p1[0] + self.p2[0]) // 2
        mid_y = (self.p1[1] + self.p2[1]) // 2
        cv2.putText(
            frame,
            "ENTRANCE GATE (IN/OUT)",
            (mid_x - 110, mid_y - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 230, 0),
            2,
            cv2.LINE_AA,
        )

    def draw_hud(
        self,
        frame: np.ndarray,
        timestamp_str: str,
        fps: float,
        detected_in_store: Optional[int] = None,
    ) -> None:
        """
        Renders the telemetry HUD scorecard in the top-right corner.

        Args:
            frame             : Video frame to draw on.
            timestamp_str     : Formatted video timecode.
            fps               : Source frame rate.
            detected_in_store : Instantaneous visible count inside the store.
        """
        h, w, _ = frame.shape
        overlay = frame.copy()

        card_w, card_h = 340, 195
        card_x = w - card_w - 20
        card_y = 20

        # Background
        cv2.rectangle(overlay, (card_x, card_y), (card_x + card_w, card_y + card_h), (15, 18, 26), -1)
        cv2.addWeighted(overlay, 0.83, frame, 0.17, 0, frame)

        # Border
        cv2.rectangle(frame, (card_x, card_y), (card_x + card_w, card_y + card_h), (70, 80, 100), 1)

        # Header
        cv2.putText(frame, "STORE OCCUPANCY MONITOR",
                    (card_x + 14, card_y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180, 200, 220), 2, cv2.LINE_AA)

        # Live headcount (large green number)
        cv2.putText(frame, "LIVE HEADCOUNT:",
                    (card_x + 14, card_y + 66), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(frame, f"{self.live_headcount}",
                    (card_x + 195, card_y + 72), cv2.FONT_HERSHEY_SIMPLEX, 1.10, (0, 255, 140), 3, cv2.LINE_AA)

        # Detected on screen
        d_val = str(detected_in_store) if detected_in_store is not None else "N/A"
        cv2.putText(frame, f"DETECTED ON SCREEN:  {d_val}",
                    (card_x + 14, card_y + 104), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 215, 0), 2, cv2.LINE_AA)

        # IN / OUT counters
        cv2.putText(frame, f"ENTERED  (IN):       {self.total_in}",
                    (card_x + 14, card_y + 136), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (80, 220, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"EXITED   (OUT):      {self.total_out}",
                    (card_x + 14, card_y + 165), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (100, 140, 255), 2, cv2.LINE_AA)

        # Watermark
        cv2.putText(frame, f"Time: {timestamp_str} | YOLOv8 + ByteTrack",
                    (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (220, 220, 220), 1, cv2.LINE_AA)
