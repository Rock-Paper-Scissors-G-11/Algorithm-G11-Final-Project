"""MediaPipe Hands wrapper used by the webcam game."""

from dataclasses import dataclass
from math import exp
import threading
import time
from typing import Protocol

import cv2
import mediapipe as mp
import numpy as np

Point = tuple[int, int]


class FrameSource(Protocol):
    def read(self) -> tuple[bool, np.ndarray | None]: ...


class HandProcessor(Protocol):
    def detect(
        self,
        frame: np.ndarray,
        draw_landmarks: bool = True,
    ) -> tuple[np.ndarray, list[list[Point]]]: ...


class LandmarkSmoother:
    """Apply low-latency exponential smoothing to one tracked hand."""

    SMOOTHING_SECONDS = 0.025
    LOST_GRACE_SECONDS = 0.08

    def __init__(self) -> None:
        self._previous: list[Point] | None = None
        self._last_seen_at: float | None = None

    def update(
        self,
        hands: list[list[Point]],
        current_time: float,
    ) -> list[list[Point]]:
        if len(hands) != 1:
            if len(hands) > 1 or (
                self._last_seen_at is not None
                and current_time - self._last_seen_at > self.LOST_GRACE_SECONDS
            ):
                self.reset()
            return hands

        current = hands[0]
        if (
            self._previous is None
            or self._last_seen_at is None
            or len(self._previous) != len(current)
            or current_time - self._last_seen_at > self.LOST_GRACE_SECONDS
        ):
            smoothed = current.copy()
        else:
            elapsed = max(0.0, current_time - self._last_seen_at)
            alpha = 1.0 - exp(-elapsed / self.SMOOTHING_SECONDS)
            smoothed = [
                (
                    round(previous[0] + alpha * (point[0] - previous[0])),
                    round(previous[1] + alpha * (point[1] - previous[1])),
                )
                for previous, point in zip(self._previous, current)
            ]

        self._previous = smoothed
        self._last_seen_at = current_time
        return [smoothed]

    def reset(self) -> None:
        self._previous = None
        self._last_seen_at = None


class HandDetector:
    """Find and draw up to two hands so the game can reject multiple hands."""

    def __init__(self) -> None:
        self._hands_module = mp.solutions.hands
        self._drawing = mp.solutions.drawing_utils
        self._tracker = self._hands_module.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._smoother = LandmarkSmoother()

    def detect(
        self,
        frame: np.ndarray,
        draw_landmarks: bool = True,
    ) -> tuple[np.ndarray, list[list[Point]]]:
        """Return the frame and pixel landmarks, optionally drawing the skeleton."""
        frame_height, frame_width = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._tracker.process(rgb_frame)

        all_landmarks: list[list[Point]] = []
        detected_hands = results.multi_hand_landmarks or []

        for hand_landmarks in detected_hands:
            points = [
                (int(point.x * frame_width), int(point.y * frame_height))
                for point in hand_landmarks.landmark
            ]
            all_landmarks.append(points)
            if draw_landmarks:
                self._drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self._hands_module.HAND_CONNECTIONS,
                )

        all_landmarks = self._smoother.update(all_landmarks, time.monotonic())
        return frame, all_landmarks

    def close(self) -> None:
        self._tracker.close()


@dataclass(frozen=True)
class TrackingSnapshot:
    sequence: int
    frame: np.ndarray | None
    landmarks: list[list[Point]]
    error: str | None = None
    detection_ms: float = 0.0
    fps: float = 0.0


class HandTrackingWorker:
    """Continuously capture and process frames away from the UI thread."""

    def __init__(
        self,
        camera: FrameSource,
        detector: HandProcessor,
    ) -> None:
        self._camera = camera
        self._detector = detector
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._snapshot = TrackingSnapshot(0, None, [])
        self._draw_landmarks = True
        self._thread = threading.Thread(
            target=self._run,
            name="hand-tracking",
            daemon=True,
        )
        self._thread.start()

    def set_draw_landmarks(self, enabled: bool) -> None:
        with self._lock:
            self._draw_landmarks = enabled

    def latest(self) -> TrackingSnapshot:
        with self._lock:
            return self._snapshot

    @property
    def is_alive(self) -> bool:
        return self._thread.is_alive()

    def close(self) -> None:
        self._stop_event.set()
        self._thread.join()

    def _publish(
        self,
        frame: np.ndarray | None,
        landmarks: list[list[Point]],
        error: str | None = None,
        detection_ms: float = 0.0,
        fps: float = 0.0,
    ) -> None:
        with self._lock:
            self._snapshot = TrackingSnapshot(
                self._snapshot.sequence + 1,
                frame,
                landmarks,
                error,
                detection_ms,
                fps,
            )

    def _run(self) -> None:
        previous_frame_at: float | None = None
        fps = 0.0
        try:
            while not self._stop_event.is_set():
                success, frame = self._camera.read()
                if not success or frame is None:
                    raise RuntimeError(
                        "The camera opened, but a video frame could not be read."
                    )
                frame = cv2.flip(frame, 1)
                with self._lock:
                    draw_landmarks = self._draw_landmarks
                detection_started_at = time.perf_counter()
                processed_frame, landmarks = self._detector.detect(
                    frame,
                    draw_landmarks=draw_landmarks,
                )
                completed_at = time.perf_counter()
                if previous_frame_at is not None:
                    frame_interval = completed_at - previous_frame_at
                    if frame_interval > 0:
                        instantaneous_fps = 1.0 / frame_interval
                        fps = (
                            instantaneous_fps
                            if fps == 0.0
                            else fps * 0.8 + instantaneous_fps * 0.2
                        )
                previous_frame_at = completed_at
                self._publish(
                    processed_frame,
                    landmarks,
                    detection_ms=(completed_at - detection_started_at) * 1000,
                    fps=fps,
                )
        except (AttributeError, OSError, RuntimeError, ValueError, cv2.error) as exc:
            if not self._stop_event.is_set():
                self._publish(None, [], str(exc))
