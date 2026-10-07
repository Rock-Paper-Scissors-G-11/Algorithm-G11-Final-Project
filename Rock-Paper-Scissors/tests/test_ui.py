import unittest
from unittest.mock import patch

import cv2
import numpy as np

from game import RockPaperScissorsGame
from navigation import GestureNavigator
from ui import HEIGHT, WIDTH, GameUI


class GameUITests(unittest.TestCase):
    def setUp(self) -> None:
        self.ui = GameUI()
        self.game = RockPaperScissorsGame()

    def test_navigation_screens_render_at_window_size(self) -> None:
        for screen in ("home", "tutorial", "settings", "loading", "error", "summary"):
            with self.subTest(screen=screen):
                canvas = self.ui.render(screen, self.game)
                self.assertEqual(canvas.shape, (HEIGHT, WIDTH, 3))
                self.assertEqual(canvas.dtype, np.uint8)

    def test_home_buttons_dispatch_their_actions(self) -> None:
        self.ui.render("home", self.game)
        self.ui.mouse_event(cv2.EVENT_LBUTTONDOWN, 250, 590, 0, None)
        self.assertEqual(self.ui.take_action(), "start")

        self.ui.render("home", self.game)
        self.ui.mouse_event(cv2.EVENT_LBUTTONDOWN, 550, 590, 0, None)
        self.assertEqual(self.ui.take_action(), "tutorial")

        self.ui.render("home", self.game)
        self.ui.mouse_event(cv2.EVENT_LBUTTONDOWN, 250, 665, 0, None)
        self.assertEqual(self.ui.take_action(), "settings")

        self.ui.render("home", self.game)
        self.ui.mouse_event(cv2.EVENT_LBUTTONDOWN, 550, 665, 0, None)
        self.assertEqual(self.ui.take_action(), "exit")

    def test_hand_hit_areas_cover_every_screen_button(self) -> None:
        screen_buttons = (
            (
                "home",
                (
                    ((287, 597), "start"),
                    ((607, 597), "tutorial"),
                    ((287, 672), "settings"),
                    ((607, 672), "exit"),
                ),
            ),
            (
                "tutorial",
                (((177, 756), "home"), ((438, 756), "start")),
            ),
            (
                "settings",
                (((844, 433), "toggle_sound"), ((177, 756), "home")),
            ),
            (
                "error",
                (((548, 600), "retry"), ((801, 600), "home")),
            ),
            (
                "game",
                (
                    ((988, 707), "back"),
                    ((1112, 707), "next_round"),
                    ((1236, 707), "finish"),
                ),
            ),
            (
                "summary",
                (
                    ((549, 732), "play_again"),
                    ((811, 732), "home"),
                    ((998, 732), "exit"),
                ),
            ),
        )
        for screen, buttons in screen_buttons:
            with self.subTest(screen=screen):
                self.ui.render(screen, self.game)
                for position, expected_action in buttons:
                    with self.subTest(action=expected_action):
                        self.assertEqual(self.ui.action_at(position), expected_action)
                        navigator = GestureNavigator()
                        self.assertIsNone(
                            navigator.update(
                                screen,
                                "Point",
                                position,
                                (WIDTH, HEIGHT),
                                self.ui.action_at,
                                0.0,
                            ).action
                        )
                        self.assertIsNone(
                            navigator.update(
                                screen,
                                "Point",
                                position,
                                (WIDTH, HEIGHT),
                                self.ui.action_at,
                                0.03,
                            ).action
                        )
                        self.assertEqual(
                            navigator.update(
                                screen,
                                "Point",
                                position,
                                (WIDTH, HEIGHT),
                                self.ui.action_at,
                                0.06,
                            ).action,
                            expected_action,
                        )

    def test_shared_hand_interaction_activates_every_rendered_button_once(self) -> None:
        screens = ("home", "tutorial", "settings", "error", "game", "summary")
        for screen in screens:
            with self.subTest(screen=screen):
                self.ui.render(screen, self.game, error_message="Camera unavailable")
                rendered_buttons = tuple(self.ui._buttons)
                self.assertTrue(rendered_buttons)
                for box, expected_action in rendered_buttons:
                    with self.subTest(action=expected_action):
                        target = (
                            (box[0] + box[2]) // 2,
                            (box[1] + box[3]) // 2,
                        )
                        self.ui.render(
                            screen,
                            self.game,
                            error_message="Camera unavailable",
                            hand_position=target,
                            navigation_gesture="Open Palm",
                        )
                        self.assertGreater(self.ui._button_hover[expected_action], 0)
                        self.assertEqual(self.ui.action_at(target), expected_action)
                        navigator = GestureNavigator()
                        self.assertIsNone(
                            navigator.update(
                                screen,
                                "Open Palm",
                                target,
                                (WIDTH, HEIGHT),
                                self.ui.action_at,
                                0.0,
                            ).action
                        )
                        states = [
                            navigator.update(
                                screen,
                                "Point",
                                target,
                                (WIDTH, HEIGHT),
                                self.ui.action_at,
                                current_time,
                            )
                            for current_time in (0.03, 0.06, 0.09)
                        ]
                        self.assertEqual(states[-1].action, expected_action)
                        self.assertEqual(navigator.interaction_state, "WAITING")

                        for current_time in (0.1, 0.3, 0.34):
                            self.assertIsNone(
                                navigator.update(
                                    screen,
                                    "Point",
                                    target,
                                    (WIDTH, HEIGHT),
                                    self.ui.action_at,
                                    current_time,
                                ).action
                            )

                        for current_time in (0.35, 0.47):
                            navigator.update(
                                screen,
                                "Unknown",
                                None,
                                (WIDTH, HEIGHT),
                                self.ui.action_at,
                                current_time,
                            )
                        self.assertEqual(navigator.interaction_state, "READY")

    def test_interaction_debug_draws_button_bounds_and_status(self) -> None:
        canvas = self.ui.render("home", self.game)
        with patch("ui._text") as draw_text:
            self.ui.draw_interaction_debug(
                canvas,
                1,
                (200, 300),
                (400, 500),
                "start",
                "PRESSED",
                12.5,
                30.0,
            )

        labels = [call.args[1] for call in draw_text.call_args_list]
        self.assertIn("Hand: DETECTED", labels)
        self.assertIn("Hand connection: ACTIVE", labels)
        self.assertIn("Detection time: 12.5 ms", labels)
        self.assertIn("Index fingertip: (200, 300)", labels)
        self.assertIn("Cursor: (400, 500)", labels)
        self.assertIn("FPS: 30.0", labels)
        self.assertIn("Hovered button: START GAME", labels)
        self.assertIn("Click: PRESSED", labels)

    def test_game_action_buttons_fit_inside_card_with_even_spacing(self) -> None:
        panel = (923, 177, 1306, 748)
        boxes = self.ui._game_action_button_boxes(panel)
        self.assertEqual(len(boxes), 3)

        for box in boxes:
            self.assertGreaterEqual(box[0] - self.ui.BUTTON_HIT_PADDING, panel[0])
            self.assertGreaterEqual(box[1] - self.ui.BUTTON_HIT_PADDING, panel[1])
            self.assertLessEqual(box[2] + self.ui.BUTTON_HIT_PADDING, panel[2])
            self.assertLessEqual(box[3] + self.ui.BUTTON_HIT_PADDING, panel[3])

        heights = {box[3] - box[1] for box in boxes}
        self.assertEqual(heights, {48})
        self.assertEqual(
            boxes[1][0] - boxes[0][2],
            boxes[2][0] - boxes[1][2],
        )
        self.assertGreaterEqual(
            boxes[1][0] - boxes[0][2],
            self.ui.BUTTON_HIT_PADDING * 2 + 1,
        )

    def test_game_action_buttons_stack_when_panel_is_narrow(self) -> None:
        panel = (100, 200, 400, 500)
        boxes = self.ui._game_action_button_boxes(panel)
        self.assertEqual(len(boxes), 3)
        self.assertTrue(all(box[1] < box[3] for box in boxes))

        self.assertEqual(len({box[0] for box in boxes}), 1)
        self.assertEqual(len({box[2] for box in boxes}), 1)
        self.assertTrue(
            all(
                boxes[index + 1][1] - boxes[index][3] == 14
                for index in range(len(boxes) - 1)
            )
        )

    def test_screen_button_hitboxes_remain_separate(self) -> None:
        for screen in ("home", "tutorial", "settings", "error", "game", "summary"):
            with self.subTest(screen=screen):
                self.ui.render(screen, self.game)
                boxes = [box for box, _ in self.ui._buttons]
                for index, first in enumerate(boxes):
                    for second in boxes[index + 1:]:
                        overlaps = (
                            first[0] <= second[2]
                            and second[0] <= first[2]
                            and first[1] <= second[3]
                            and second[1] <= first[3]
                        )
                        self.assertFalse(overlaps, (first, second))

    def test_result_screen_keeps_secondary_actions_available(self) -> None:
        self.game.start_round(1.0)
        self.game.update("Unknown", 0, 4.0)
        with patch("game.random.choice", return_value="Scissors"):
            self.game.update("Rock", 1, 4.1)

        self.ui.render("game", self.game)
        main_menu_box, next_round_box, end_game_box = (
            self.ui._game_action_button_boxes((923, 177, 1306, 748))
        )
        centers = tuple(
            ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)
            for box in (main_menu_box, next_round_box, end_game_box)
        )
        self.assertEqual(self.ui.action_at(centers[0]), "back")
        self.assertIsNone(self.ui.action_at(centers[1]))
        self.assertEqual(self.ui.action_at(centers[2]), "finish")

    def test_hit_area_expands_without_moving_button_or_hover_highlights(self) -> None:
        box = (136, 566, 438, 628)
        canvas = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        self.ui._hand_position = (130, 590)
        self.ui._button(canvas, box, "Start Game", "start", primary=True)

        self.assertEqual(self.ui.action_at((130, 590)), "start")
        self.assertEqual(self.ui.action_at((129, 590)), "start")
        self.assertIsNone(self.ui.action_at((127, 590)))
        self.assertEqual(
            tuple(canvas[590, 200]),
            self.ui._mix_color((255, 220, 0), (255, 245, 72), 0.65),
        )
        self.assertEqual(self.ui._buttons[0][0], (128, 558, 446, 636))

    def test_button_hover_fades_without_changing_button_bounds(self) -> None:
        self.ui._render_delta = 1 / 60
        canvas = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        box = (136, 566, 438, 628)
        self.ui._hand_position = (200, 590)
        self.ui._button(canvas, box, "Start Game", "start", primary=True)
        active_amount = self.ui._button_hover["start"]

        self.ui._hand_position = None
        self.ui._button(canvas, box, "Start Game", "start", primary=True)
        fading_amount = self.ui._button_hover["start"]

        self.assertLess(fading_amount, active_amount)
        self.assertGreater(fading_amount, 0)
        self.assertEqual(self.ui._buttons[0][0], (128, 558, 446, 636))

    def test_button_positions_remain_fixed_across_renders(self) -> None:
        self.ui.render("game", self.game)
        initial_buttons = tuple(self.ui._buttons)

        for _ in range(3):
            self.ui.render("game", self.game)
            self.assertEqual(tuple(self.ui._buttons), initial_buttons)

    def test_screen_transition_does_not_overlay_old_button_positions(self) -> None:
        self.ui.render("home", self.game)
        self.ui.render("tutorial", self.game)

        np.testing.assert_array_equal(
            self.ui._transition_canvas,
            self.ui._background(),
        )

    def test_scaled_viewport_preserves_mouse_button_selection(self) -> None:
        self.ui.render("home", self.game)
        logical_frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        displayed = self.ui.fit_to_viewport(logical_frame, (680, 430))
        self.assertEqual(displayed.shape, (430, 680, 3))

        self.ui.mouse_event(cv2.EVENT_LBUTTONDOWN, 125, 295, 0, None)
        self.assertEqual(self.ui.take_action(), "start")

    def test_letterboxed_viewport_ignores_clicks_outside_the_ui(self) -> None:
        self.ui.render("home", self.game)
        logical_frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        displayed = self.ui.fit_to_viewport(logical_frame, (800, 400))
        self.assertEqual(displayed.shape, (400, 800, 3))
        self.ui.mouse_event(cv2.EVENT_LBUTTONDOWN, 10, 200, 0, None)
        self.assertIsNone(self.ui.take_action())

        offset_x, offset_y = self.ui._viewport_offset
        center_x = round(offset_x + 287 * self.ui._viewport_scale)
        center_y = round(offset_y + 597 * self.ui._viewport_scale)
        self.ui.mouse_event(
            cv2.EVENT_LBUTTONDOWN,
            center_x,
            center_y,
            0,
            None,
        )
        self.assertEqual(self.ui.take_action(), "start")

    def test_settings_slider_maps_click_to_master_volume(self) -> None:
        self.ui.render("settings", self.game)
        self.ui.mouse_event(cv2.EVENT_LBUTTONDOWN, 680, 567, 0, None)
        self.assertEqual(self.ui.take_action(), "volume_50")

    def test_game_screen_renders_live_camera_frame(self) -> None:
        camera_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        canvas = self.ui.render(
            "game",
            self.game,
            camera_frame,
            camera_active=True,
            hands_detected=0,
        )

        self.assertEqual(canvas.shape, (HEIGHT, WIDTH, 3))
        np.testing.assert_array_equal(canvas[455, 475], (0, 0, 0))

    def test_camera_display_preserves_entire_source_aspect_ratio(self) -> None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:20] = (0, 0, 255)
        frame[-20:] = (0, 255, 0)
        canvas = np.full((HEIGHT, WIDTH, 3), (42, 23, 15), dtype=np.uint8)
        self.ui._fit_frame(canvas, frame, (78, 225, 871, 707))

        np.testing.assert_array_equal(canvas[230, 475], (0, 0, 255))
        np.testing.assert_array_equal(canvas[700, 475], (0, 255, 0))
        np.testing.assert_array_equal(canvas[465, 100], (42, 23, 15))

    def test_home_screen_shows_live_hand_preview(self) -> None:
        camera_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        canvas = self.ui.render("home", self.game, camera_frame)

        np.testing.assert_array_equal(canvas[288, 700], (0, 0, 0))

    def test_round_result_renders_both_choices(self) -> None:
        self.game.start_round(1.0)
        self.game.update("Unknown", 0, 4.0)
        with patch("game.random.choice", return_value="Scissors"):
            self.game.update("Rock", 1, 4.1)

        canvas = self.ui.render(
            "game",
            self.game,
            camera_active=True,
            hands_detected=1,
            gesture="Rock",
        )

        self.assertEqual(canvas.shape, (HEIGHT, WIDTH, 3))

    def test_playing_screen_explains_short_gesture_stabilization(self) -> None:
        self.game.start_round(1.0)
        self.game.update("Unknown", 0, 4.0)
        with patch("ui._text") as draw_text:
            self.ui.render(
                "game",
                self.game,
                hands_detected=1,
                gesture_pending=True,
            )

        self.assertTrue(
            any(
                call.args[1] == "Hold still to lock in your choice."
                for call in draw_text.call_args_list
            )
        )

    def test_game_screen_shows_live_hand_feedback(self) -> None:
        self.game.start_round(1.0)
        self.game.update("Unknown", 0, 4.0)
        with patch("ui._text") as draw_text:
            self.ui.render(
                "game",
                self.game,
                hands_detected=1,
                gesture="Paper",
                gesture_hint="Hand detected",
            )

        labels = [call.args[1] for call in draw_text.call_args_list]
        self.assertIn("HAND DETECTED", labels)
        self.assertIn("Paper", labels)

    def test_hand_pointer_is_rendered_for_aiming_before_point_gesture(self) -> None:
        position = (250, 590)
        with patch.object(self.ui, "_draw_hand_pointer") as draw_pointer:
            self.ui.render(
                "home",
                self.game,
                hand_position=position,
                navigation_gesture="Open Palm",
            )

        draw_pointer.assert_called_once_with(unittest.mock.ANY, position)

    def test_shoot_feedback_is_prominent_at_play_transition(self) -> None:
        self.game.start_round(1.0)
        self.game.update("Unknown", 0, 4.0)
        with patch("ui._center_text") as draw_centered:
            self.ui.render(
                "game",
                self.game,
                go_feedback=True,
            )

        self.assertTrue(
            any(call.args[1] == "SHOOT!" for call in draw_centered.call_args_list)
        )


if __name__ == "__main__":
    unittest.main()
