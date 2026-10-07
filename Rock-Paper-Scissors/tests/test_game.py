import unittest
from unittest.mock import patch

from game import GameState, RockPaperScissorsGame, determine_winner
from gesture import GestureStabilizer, detect_gesture, hand_size_ratio
from main import advance_round_gesture


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
        if finger_index in open_fingers:
            points[pip] = (x, 120)
            points[tip] = (x, 80)
        else:
            points[pip] = (x, 120)
            points[tip] = (x, 145)

    if thumb_open:
        points[3] = (120, 150)
        points[4] = (80, 150)
    else:
        points[3] = (130, 150)
        points[4] = (135, 150)
    return points


class WinnerTests(unittest.TestCase):
    def test_all_winner_combinations(self) -> None:
        self.assertEqual(determine_winner("Rock", "Scissors"), "Player Wins")
        self.assertEqual(determine_winner("Scissors", "Paper"), "Player Wins")
        self.assertEqual(determine_winner("Paper", "Rock"), "Player Wins")
        self.assertEqual(determine_winner("Scissors", "Rock"), "Computer Wins")
        self.assertEqual(determine_winner("Paper", "Scissors"), "Computer Wins")
        self.assertEqual(determine_winner("Rock", "Paper"), "Computer Wins")
        self.assertEqual(determine_winner("Paper", "Paper"), "Draw")

    def test_invalid_choice_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            determine_winner("Unknown", "Rock")


class GestureTests(unittest.TestCase):
    def test_recognizes_rock_paper_and_scissors(self) -> None:
        self.assertEqual(detect_gesture(make_hand(set())), "Rock")
        self.assertEqual(detect_gesture(make_hand({0, 1, 2, 3}, True)), "Paper")
        self.assertEqual(detect_gesture(make_hand({0, 1})), "Scissors")

    def test_gesture_classification_tolerates_rotated_hands(self) -> None:
        for fingers, thumb_open, expected in (
            (set(), False, "Rock"),
            ({0, 1, 2, 3}, True, "Paper"),
            ({0, 1}, False, "Scissors"),
        ):
            points = make_hand(fingers, thumb_open)
            center_x, center_y = 180, 150
            rotated = []
            for x, y in points:
                dx, dy = x - center_x, y - center_y
                rotated.append(
                    (
                        round(center_x + dx * 0.866 - dy * 0.5),
                        round(center_y + dx * 0.5 + dy * 0.866),
                    )
                )
            with self.subTest(gesture=expected):
                self.assertEqual(detect_gesture(rotated), expected)

    def test_returns_unknown_for_other_or_incomplete_poses(self) -> None:
        self.assertEqual(detect_gesture(make_hand({0})), "Unknown")
        self.assertEqual(detect_gesture([(0, 0)]), "Unknown")


class HandSizeTests(unittest.TestCase):
    def test_hand_size_ratio_is_scale_invariant(self) -> None:
        points = make_hand({0, 1}, False)
        doubled = [(x * 2, y * 2) for x, y in points]
        first = hand_size_ratio(points, (640, 480))
        second = hand_size_ratio(doubled, (1280, 960))
        self.assertAlmostEqual(first, second)

    def test_hand_size_ratio_rejects_invalid_frame_dimensions(self) -> None:
        with self.assertRaises(ValueError):
            hand_size_ratio(make_hand(set()), (0, 480))


class GestureStabilizerTests(unittest.TestCase):
    def test_all_choices_stabilize_after_two_matching_frames(self) -> None:
        for choice in ("Rock", "Paper", "Scissors"):
            with self.subTest(choice=choice):
                stabilizer = GestureStabilizer()
                self.assertEqual(stabilizer.update(choice, 0.0), "Unknown")
                self.assertEqual(stabilizer.update(choice, 0.033), choice)

    def test_single_frame_noise_does_not_switch_stable_gesture(self) -> None:
        stabilizer = GestureStabilizer()
        self.assertEqual(stabilizer.update("Rock", 0.0), "Unknown")
        self.assertEqual(stabilizer.update("Rock", 0.033), "Rock")
        self.assertEqual(stabilizer.update("Paper", 0.066), "Rock")
        self.assertEqual(stabilizer.update("Rock", 0.099), "Rock")

    def test_gesture_changes_are_accepted_in_two_frames(self) -> None:
        stabilizer = GestureStabilizer()
        self.assertEqual(stabilizer.update("Rock", 0.0), "Unknown")
        self.assertEqual(stabilizer.update("Rock", 0.033), "Rock")
        self.assertEqual(stabilizer.update("Paper", 0.066), "Rock")
        self.assertEqual(stabilizer.update("Paper", 0.099), "Paper")
        self.assertEqual(stabilizer.update("Scissors", 0.132), "Paper")
        self.assertEqual(stabilizer.update("Scissors", 0.165), "Scissors")

    def test_camera_loss_clears_stable_pose_after_short_grace_period(self) -> None:
        stabilizer = GestureStabilizer()
        stabilizer.update("Paper", 0.0)
        stabilizer.update("Paper", 0.033)
        self.assertEqual(stabilizer.update("Unknown", 0.05), "Unknown")
        self.assertEqual(stabilizer.update("Scissors", 0.2), "Unknown")
        self.assertEqual(stabilizer.update("Scissors", 0.233), "Scissors")

    def test_stabilized_choice_completes_one_round_without_changing_rules(self) -> None:
        cases = (
            ("Rock", "Scissors", 1, 0, 0),
            ("Paper", "Scissors", 0, 1, 0),
            ("Scissors", "Scissors", 0, 0, 1),
        )
        for player_choice, computer_choice, wins, losses, draws in cases:
            with self.subTest(player_choice=player_choice):
                game = RockPaperScissorsGame()
                stabilizer = GestureStabilizer()
                game.start_round(0.0)
                game.update("Unknown", 0, 3.0)
                self.assertEqual(game.state, GameState.PLAYING)

                first_frame = stabilizer.update(player_choice, 3.01)
                self.assertEqual(first_frame, "Unknown")
                game.update(first_frame, 1, 3.01)
                stable_choice = stabilizer.update(player_choice, 3.07)
                with patch("game.random.choice", return_value=computer_choice):
                    game.update(stable_choice, 1, 3.07)

                self.assertEqual(game.state, GameState.RESULT)
                self.assertEqual(
                    (game.player_score, game.computer_score, game.draw_score),
                    (wins, losses, draws),
                )

    def test_stable_pose_is_ready_at_shoot_and_captured_next_frame(self) -> None:
        game = RockPaperScissorsGame()
        stabilizer = GestureStabilizer()
        game.start_round(0.0)
        pending_choice: str | None = None
        for current_time in (2.85, 2.9, 2.95):
            pending_choice, _, _ = advance_round_gesture(
                game,
                stabilizer,
                "Scissors",
                1,
                current_time,
                pending_choice,
            )
        self.assertIsNone(pending_choice)

        pending_choice, _, _ = advance_round_gesture(
            game,
            stabilizer,
            "Scissors",
            1,
            3.0,
            pending_choice,
        )
        self.assertEqual(game.state, GameState.PLAYING)
        self.assertEqual(pending_choice, "Scissors")
        with patch("game.random.choice", return_value="Paper"):
            pending_choice, _, _ = advance_round_gesture(
                game,
                stabilizer,
                "Unknown",
                0,
                3.01,
                pending_choice,
            )

        self.assertIsNone(pending_choice)
        self.assertEqual(game.state, GameState.RESULT)
        self.assertEqual(game.player_choice, "Scissors")
        self.assertEqual(game.computer_choice, "Paper")


class RoundTests(unittest.TestCase):
    def test_countdown_single_result_scoring_and_round_delay(self) -> None:
        game = RockPaperScissorsGame()
        game.start_round(10.0)
        self.assertEqual(game.state, GameState.COUNTDOWN)
        self.assertEqual(game.countdown_number(10.0), 3)

        game.update("Rock", 1, 13.0)
        self.assertEqual(game.state, GameState.PLAYING)
        with patch("game.random.choice", return_value="Scissors"):
            game.update("Rock", 1, 13.1)
            game.update("Paper", 1, 13.2)

        self.assertEqual(game.state, GameState.RESULT)
        self.assertEqual(game.player_score, 1)
        self.assertEqual(game.player_choice, "Rock")
        game.update("Paper", 1, 16.0)
        self.assertEqual(game.state, GameState.COUNTDOWN)
        self.assertEqual(game.countdown_number(16.0), 3)

    def test_match_ends_after_three_player_wins_and_not_before(self) -> None:
        game = RockPaperScissorsGame()
        with patch("game.random.choice", return_value="Scissors"):
            for round_number in range(3):
                round_start = round_number * 10.0
                game.start_round(round_start)
                game.update("Unknown", 0, round_start + game.COUNTDOWN_SECONDS)
                game.update("Rock", 1, round_start + game.COUNTDOWN_SECONDS + 0.1)

                if round_number < 2:
                    self.assertEqual(game.match_winner, None)
                    game.update(
                        "Unknown",
                        0,
                        round_start
                        + game.COUNTDOWN_SECONDS
                        + 0.1
                        + game.RESULT_SECONDS
                        + 0.02,
                    )
                    self.assertEqual(game.state, GameState.COUNTDOWN)
                else:
                    self.assertEqual(game.match_winner, "Player")
                    game.update(
                        "Unknown",
                        0,
                        round_start
                        + game.COUNTDOWN_SECONDS
                        + 0.1
                        + game.RESULT_SECONDS
                        + 0.02,
                    )
                    self.assertEqual(game.state, GameState.GAME_OVER)
                    self.assertEqual(game.status_message, "You are the winner!")

        self.assertEqual(game.player_score, 3)
        game.start_round(100.0)
        self.assertEqual(game.state, GameState.GAME_OVER)

    def test_draw_does_not_count_toward_three_wins(self) -> None:
        game = RockPaperScissorsGame()
        with patch("game.random.choice", return_value="Rock"):
            game.start_round(0.0)
            game.update("Unknown", 0, 3.0)
            game.update("Rock", 1, 3.1)
            game.update("Unknown", 0, 3.1 + game.RESULT_SECONDS + 0.02)

        self.assertEqual(game.draw_score, 1)
        self.assertEqual(game.player_score, 0)
        self.assertEqual(game.computer_score, 0)
        self.assertEqual(game.state, GameState.COUNTDOWN)

    def test_match_ends_after_three_computer_wins(self) -> None:
        game = RockPaperScissorsGame()
        with patch("game.random.choice", return_value="Paper"):
            for round_number in range(3):
                round_start = round_number * 10.0
                game.start_round(round_start)
                game.update("Unknown", 0, round_start + game.COUNTDOWN_SECONDS)
                game.update("Rock", 1, round_start + game.COUNTDOWN_SECONDS + 0.1)
                game.update(
                    "Unknown",
                    0,
                    round_start
                    + game.COUNTDOWN_SECONDS
                    + 0.1
                    + game.RESULT_SECONDS
                    + 0.02,
                )

        self.assertEqual(game.computer_score, 3)
        self.assertEqual(game.match_winner, "Computer")
        self.assertEqual(game.state, GameState.GAME_OVER)
        self.assertEqual(game.status_message, "The computer is the winner!")

    def test_reset_clears_score_and_returns_to_waiting(self) -> None:
        game = RockPaperScissorsGame()
        game.player_score = 2
        game.computer_score = 1
        game.draw_score = 3
        game.reset_score()
        self.assertEqual(
            (game.player_score, game.computer_score, game.draw_score),
            (0, 0, 0),
        )
        self.assertEqual(game.state, GameState.WAITING)


if __name__ == "__main__":
    unittest.main()
