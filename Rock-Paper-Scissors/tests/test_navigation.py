import unittest
from collections.abc import Callable

from game import CHOICES
from gesture import detect_gesture, detect_navigation_gesture
from navigation import GestureNavigator
from ui import HEIGHT, WIDTH


def make_hand(open_fingers: set[int], thumb_open: bool = False) -> list[tuple[int, int]]:
    points = [(100, 100) for _ in range(21)]
    points[5] = (140, 150)
    points[17] = (220, 150)
    finger_landmarks = (
        (5, 6, 8),
        (9, 10, 12),
        (13, 14, 16),
        (17, 18, 20),
    )
    for finger_index, (mcp, pip, tip) in enumerate(finger_landmarks):
        x = 150 + finger_index * 20
        points[mcp] = (x, 160)
        points[pip] = (x, 120)
        points[tip] = (x, 80 if finger_index in open_fingers else 145)
    if thumb_open:
        points[3] = (120, 150)
        points[4] = (80, 150)
    else:
        points[3] = (130, 150)
        points[4] = (135, 150)
    points[0] = (180, 220)
    return points


class NavigationGestureTests(unittest.TestCase):
    def test_navigation_gestures_are_distinct_from_rps_choices(self) -> None:
        fist = make_hand(set())
        point = make_hand({0})
        open_palm = make_hand({0, 1, 2, 3}, True)
        thumbs_up = make_hand(set())
        thumbs_up[3] = (130, 110)
        thumbs_up[4] = (130, 75)

        self.assertEqual(detect_navigation_gesture(fist), "Fist")
        self.assertEqual(detect_navigation_gesture(point), "Point")
        self.assertEqual(detect_navigation_gesture(open_palm), "Open Palm")
        self.assertEqual(detect_navigation_gesture(thumbs_up), "Thumbs Up")
        self.assertEqual(detect_gesture(fist), "Rock")
        self.assertEqual(detect_gesture(open_palm), "Paper")
        self.assertEqual(CHOICES, ("Rock", "Paper", "Scissors"))


class GestureNavigatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.navigator = GestureNavigator()

    @staticmethod
    def action_at(position: tuple[int, int]) -> str | None:
        x, y = position
        if 136 <= x <= 438 and 566 <= y <= 628:
            return "start"
        return None

    def update(
        self,
        gesture: str,
        point: tuple[int, int] | None,
        action_at: Callable[[tuple[int, int]], str | None],
        now: float,
        screen: str = "home",
    ):
        return self.navigator.update(
            screen,
            gesture,
            point,
            (WIDTH, HEIGHT),
            action_at,
            now,
        )

    def test_point_click_confirms_in_three_frames_and_requires_release(self) -> None:
        first = self.update("Point", (200, 590), self.action_at, 0.0)
        self.assertIsNone(first.action)
        self.assertEqual(first.cursor, (200, 590))
        self.assertTrue(first.hover_changed)

        second = self.update("Point", (200, 590), self.action_at, 0.03)
        self.assertIsNone(second.action)
        selected = self.update("Point", (200, 590), self.action_at, 0.06)
        self.assertEqual(selected.action, "start")

        self.update("Unknown", None, self.action_at, 0.07)
        self.assertIsNone(
            self.update("Point", (200, 590), self.action_at, 0.08).action
        )
        during_cooldown = self.update("Point", (200, 590), self.action_at, 0.3)
        self.assertIsNone(during_cooldown.action)
        self.update("Unknown", None, self.action_at, 0.31)
        self.update("Unknown", None, self.action_at, 0.43)
        self.assertIsNone(self.update("Point", (200, 590), self.action_at, 0.44).action)
        self.assertIsNone(self.update("Point", (200, 590), self.action_at, 0.47).action)
        selected_again = self.update("Point", (200, 590), self.action_at, 0.5)
        self.assertEqual(selected_again.action, "start")

        still_held = self.update("Point", (200, 590), self.action_at, 0.53)
        self.assertIsNone(still_held.action)
        self.update("Unknown", None, self.action_at, 0.54)
        self.assertIsNone(
            self.update("Point", (200, 590), self.action_at, 0.55).action
        )
        self.update("Unknown", None, self.action_at, 0.56)
        self.update("Unknown", None, self.action_at, 0.68)
        selected_after_release = self.update(
            "Point", (200, 590), self.action_at, 0.69
        )
        self.assertIsNone(selected_after_release.action)
        self.assertIsNone(self.update("Point", (200, 590), self.action_at, 0.72).action)
        selected_after_dwell = self.update(
            "Point", (200, 590), self.action_at, 0.75
        )
        self.assertEqual(selected_after_dwell.action, "start")

    def test_point_cursor_uses_light_smoothing(self) -> None:
        self.update("Point", (200, 590), self.action_at, 0.0)
        moved = self.update("Point", (400, 590), self.action_at, 0.1)
        self.assertIsNotNone(moved.cursor)
        self.assertGreater(moved.cursor[0], 200)
        self.assertGreater(moved.cursor[0], 300)
        self.assertLess(moved.cursor[0], 400)

    def test_cursor_and_hover_follow_index_tip_before_point_activation(self) -> None:
        target = (200, 590)
        hovering = self.update("Open Palm", target, self.action_at, 0.0)
        self.assertEqual(hovering.cursor, target)
        self.assertTrue(hovering.hover_changed)
        self.assertIsNone(hovering.action)

        pointing = self.update("Point", target, self.action_at, 0.05)
        self.assertFalse(pointing.hover_changed)
        self.assertIsNone(pointing.action)
        self.assertIsNone(self.update("Point", target, self.action_at, 0.08).action)
        self.assertEqual(self.update("Point", target, self.action_at, 0.11).action, "start")

    def test_moving_across_button_does_not_complete_click_confirmation(self) -> None:
        self.update("Point", (200, 590), self.action_at, 0.0)
        self.update("Point", (200, 590), self.action_at, 0.03)
        moving = self.update("Point", (400, 590), self.action_at, 0.06)

        self.assertIsNone(moving.action)

    def test_click_confirmed_once_only_after_release_and_cooldown(self) -> None:
        for now in (0.0, 0.03):
            self.update("Point", (200, 590), self.action_at, now)
        self.assertEqual(
            self.update("Point", (200, 590), self.action_at, 0.06).action,
            "start",
        )
        for now in (0.07, 0.2, 0.31, 0.34):
            self.assertIsNone(
                self.update("Point", (200, 590), self.action_at, now).action
            )

    def test_camera_edge_coordinates_map_to_matching_ui_edges(self) -> None:
        left = self.update("Open Palm", (0, 100), self.action_at, 0.0)
        self.assertEqual(left.cursor, (0, round(100 * HEIGHT / HEIGHT)))

        self.navigator.reset()
        right = self.update("Open Palm", (WIDTH, 100), self.action_at, 0.0)
        self.assertEqual(right.cursor, (WIDTH - 1, round(100 * HEIGHT / HEIGHT)))

        self.navigator.reset()
        left_camera = self.navigator.update(
            "home",
            "Open Palm",
            (0, 0),
            (640, 480),
            self.action_at,
            0.0,
        )
        self.navigator.reset()
        right_camera = self.navigator.update(
            "home",
            "Open Palm",
            (640, 480),
            (640, 480),
            self.action_at,
            0.1,
        )
        self.assertEqual(left_camera.cursor, (0, 0))
        self.assertEqual(right_camera.cursor, (WIDTH - 1, HEIGHT - 1))

    def test_moving_off_a_button_rearms_selection(self) -> None:
        self.update("Point", (200, 590), self.action_at, 0.0)
        self.update("Point", (200, 590), self.action_at, 0.03)
        selected = self.update("Point", (200, 590), self.action_at, 0.06)
        self.assertEqual(selected.action, "start")

        away = self.update("Point", (1000, 590), self.action_at, 0.1)
        self.assertIsNone(away.action)
        self.update("Point", (1000, 590), self.action_at, 0.22)
        self.update("Point", (200, 590), self.action_at, 0.25)
        returned = self.update("Point", (200, 590), self.action_at, 0.28)
        self.assertIsNone(returned.action)
        self.assertIsNone(self.update("Point", (200, 590), self.action_at, 0.31).action)
        self.assertEqual(self.navigator.interaction_state, "WAITING")
        self.update("Unknown", None, self.action_at, 0.32)
        self.update("Unknown", None, self.action_at, 0.44)
        self.update("Point", (200, 590), self.action_at, 0.45)
        self.update("Point", (200, 590), self.action_at, 0.48)
        selected_again = self.update("Point", (200, 590), self.action_at, 0.56)
        self.assertEqual(selected_again.action, "start")

    def test_fist_requires_stability_before_back(self) -> None:
        self.assertIsNone(
            self.update("Fist", None, self.action_at, 0.0, "tutorial").action
        )
        self.assertEqual(
            self.update("Fist", None, self.action_at, 0.65, "tutorial").action,
            "back",
        )

    def test_cursor_expires_after_pointing_stops(self) -> None:
        self.update("Point", (200, 590), self.action_at, 1.0)
        self.assertIsNone(
            self.update("Unknown", None, self.action_at, 1.16).cursor
        )

    def test_cursor_survives_short_loss_and_snaps_after_genuine_loss(self) -> None:
        self.update("Open Palm", (200, 590), self.action_at, 0.0)

        during_grace = self.update("Unknown", None, self.action_at, 0.1)
        self.assertEqual(during_grace.cursor, (200, 590))

        self.update("Unknown", None, self.action_at, 0.2)
        recovered = self.update("Open Palm", (700, 300), self.action_at, 0.21)
        self.assertEqual(recovered.cursor, (700, 300))

    def test_cursor_snaps_to_new_position_after_screen_change(self) -> None:
        self.update("Open Palm", (200, 590), self.action_at, 0.0)

        changed_screen = self.update(
            "Open Palm",
            (700, 300),
            self.action_at,
            0.03,
            screen="tutorial",
        )

        self.assertEqual(changed_screen.cursor, (700, 300))

    def test_small_cursor_jitter_does_not_move_the_cursor(self) -> None:
        self.update("Open Palm", (200, 590), self.action_at, 0.0)

        jittered = self.update("Open Palm", (201, 591), self.action_at, 0.03)

        self.assertEqual(jittered.cursor, (200, 590))
