import threading
import time
import unittest

import numpy as np

from hand_detector import HandTrackingWorker, LandmarkSmoother


class FakeCamera:
    def __init__(self) -> None:
        self.frame_number = 0

    def read(self) -> tuple[bool, np.ndarray]:
        time.sleep(0.002)
        self.frame_number += 1
        frame = np.full((4, 4, 3), self.frame_number % 255, dtype=np.uint8)
        return True, frame


class FakeDetector:
    def __init__(self) -> None:
        self.processed = threading.Event()

    def detect(
        self,
        frame: np.ndarray,
        draw_landmarks: bool = True,
    ) -> tuple[np.ndarray, list[list[tuple[int, int]]]]:
        del draw_landmarks
        self.processed.set()
        return frame, [[(1, 0)]]


class LandmarkSmootherTests(unittest.TestCase):
    def test_smoothing_reduces_jitter_without_a_long_lag(self) -> None:
        smoother = LandmarkSmoother()
        origin = [(0, 0)] * 21
        moved = [(100, 60)] * 21

        self.assertEqual(smoother.update([origin], 0.0), [origin])
        result = smoother.update([moved], 1 / 30)[0]

        self.assertGreater(result[0][0], 70)
        self.assertLess(result[0][0], 80)
        self.assertGreater(result[0][1], 40)
        self.assertLess(result[0][1], 50)

    def test_missed_frame_is_not_fabricated_and_short_dropout_recovers(self) -> None:
        smoother = LandmarkSmoother()
        before = [(10, 10)] * 21
        after = [(50, 50)] * 21
        smoother.update([before], 0.0)

        self.assertEqual(smoother.update([], 0.03), [])
        recovered = smoother.update([after], 0.05)[0]
        self.assertGreater(recovered[0][0], 10)
        self.assertLess(recovered[0][0], 50)

    def test_long_dropout_resets_smoothing(self) -> None:
        smoother = LandmarkSmoother()
        smoother.update([[(10, 10)] * 21], 0.0)
        target = [(50, 50)] * 21

        self.assertEqual(smoother.update([target], 0.2), [target])


class HandTrackingWorkerTests(unittest.TestCase):
    def test_worker_publishes_frames_and_stops_cleanly(self) -> None:
        camera = FakeCamera()
        detector = FakeDetector()
        worker = HandTrackingWorker(camera, detector)
        try:
            self.assertTrue(detector.processed.wait(1.0))
            deadline = time.monotonic() + 1.0
            snapshot = worker.latest()
            while snapshot.sequence < 2 and time.monotonic() < deadline:
                threading.Event().wait(0.001)
                snapshot = worker.latest()

            self.assertGreater(snapshot.sequence, 0)
            self.assertIsNotNone(snapshot.frame)
            self.assertIsNone(snapshot.error)
            self.assertGreater(snapshot.detection_ms, 0)
            self.assertGreater(snapshot.fps, 0)
        finally:
            worker.close()
        self.assertFalse(worker.is_alive)


if __name__ == "__main__":
    unittest.main()
