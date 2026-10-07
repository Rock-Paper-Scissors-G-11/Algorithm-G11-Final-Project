"""Classify a hand's landmarks as a Rock, Paper, or Scissors gesture."""

from collections import deque
from math import hypot
from typing import Sequence

Point = tuple[int, int]

CHOICES = ("Rock", "Paper", "Scissors")

FINGER_JOINTS = (
    (5, 6, 8),    # Index MCP, PIP, and tip
    (9, 10, 12),  # Middle MCP, PIP, and tip
    (13, 14, 16),  # Ring MCP, PIP, and tip
    (17, 18, 20),  # Pinky MCP, PIP, and tip
)


def _finger_is_open(
    points: Sequence[Point],
    mcp_id: int,
    pip_id: int,
    tip_id: int,
) -> bool:
    """Check extension along the finger axis, independent of hand rotation."""
    mcp = points[mcp_id]
    pip = points[pip_id]
    tip = points[tip_id]
    axis = (pip[0] - mcp[0], pip[1] - mcp[1])
    beyond_pip = (tip[0] - pip[0], tip[1] - pip[1])
    return axis[0] * beyond_pip[0] + axis[1] * beyond_pip[1] > 0


def _thumb_is_open(points: Sequence[Point]) -> bool:
    thumb_tip = points[4]
    thumb_ip = points[3]
    index_mcp = points[5]
    palm_width = hypot(
        points[5][0] - points[17][0],
        points[5][1] - points[17][1],
    )
    tip_distance = hypot(
        thumb_tip[0] - index_mcp[0],
        thumb_tip[1] - index_mcp[1],
    )
    joint_distance = hypot(
        thumb_ip[0] - index_mcp[0],
        thumb_ip[1] - index_mcp[1],
    )
    return tip_distance > joint_distance + palm_width * 0.25


def detect_gesture(landmarks: Sequence[Point]) -> str:
    """Return Rock, Paper, Scissors, or Unknown for 21 pixel-coordinate landmarks."""
    if len(landmarks) < 21:
        return "Unknown"

    fingers_open = [
        _finger_is_open(landmarks, mcp_id, pip_id, tip_id)
        for mcp_id, pip_id, tip_id in FINGER_JOINTS
    ]
    thumb_open = _thumb_is_open(landmarks)

    if not any(fingers_open) and not thumb_open:
        return "Rock"
    if all(fingers_open) and thumb_open:
        return "Paper"
    if fingers_open[0] and fingers_open[1] and not any(fingers_open[2:]):
        return "Scissors"
    return "Unknown"


def hand_size_ratio(
    landmarks: Sequence[Point],
    frame_size: tuple[int, int],
) -> float:
    """Return hand landmark bounding-box diagonal relative to the frame diagonal."""
    if not landmarks:
        return 0.0
    frame_width, frame_height = frame_size
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("Frame dimensions must be positive.")
    xs = [point[0] for point in landmarks]
    ys = [point[1] for point in landmarks]
    hand_diagonal = hypot(max(xs) - min(xs), max(ys) - min(ys))
    return hand_diagonal / hypot(frame_width, frame_height)


def detect_navigation_gesture(landmarks: Sequence[Point]) -> str:
    """Classify menu-only pointing, confirmation, and back gestures."""
    if len(landmarks) < 21:
        return "Unknown"

    fingers_open = [
        _finger_is_open(landmarks, mcp_id, pip_id, tip_id)
        for mcp_id, pip_id, tip_id in FINGER_JOINTS
    ]
    if fingers_open[0] and not any(fingers_open[1:]):
        return "Point"
    if all(fingers_open) and _thumb_is_open(landmarks):
        return "Open Palm"
    if not any(fingers_open) and landmarks[4][1] < landmarks[3][1]:
        return "Thumbs Up"
    if not any(fingers_open) and not _thumb_is_open(landmarks):
        return "Fist"
    return "Unknown"


class GestureStabilizer:
    """Follow gesture changes quickly while rejecting single-frame noise."""

    LOST_GRACE_SECONDS = 0.08

    def __init__(self) -> None:
        self._history: deque[str] = deque(maxlen=3)
        self._stable_gesture = "Unknown"
        self._last_match_at: float | None = None

    @property
    def candidate(self) -> str:
        return self._stable_gesture

    def update(self, gesture: str, current_time: float) -> str:
        if (
            self._last_match_at is not None
            and current_time - self._last_match_at > self.LOST_GRACE_SECONDS
        ):
            self.reset()

        if gesture not in CHOICES:
            return "Unknown"

        self._history.append(gesture)
        self._last_match_at = current_time
        for choice in CHOICES:
            if self._history.count(choice) >= 2:
                self._stable_gesture = choice
                break
        return self._stable_gesture

    def reset(self) -> None:
        self._history.clear()
        self._stable_gesture = "Unknown"
        self._last_match_at = None
