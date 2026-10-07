"""Rock Paper Scissors rules, score, and round state management."""

from enum import Enum, auto
from math import ceil
import random

CHOICES = ("Rock", "Paper", "Scissors")

class GameState(Enum):
    WAITING = auto()
    COUNTDOWN = auto()
    PLAYING = auto()
    RESULT = auto()
    GAME_OVER = auto()


def determine_winner(player_choice: str, computer_choice: str) -> str:
    """Compare two valid choices and return the round result."""
    if player_choice not in CHOICES or computer_choice not in CHOICES:
        raise ValueError("Both choices must be Rock, Paper, or Scissors.")
    if player_choice == computer_choice:
        return "Draw"

    winning_pairs = {
        ("Rock", "Scissors"),
        ("Scissors", "Paper"),
        ("Paper", "Rock"),
    }
    if (player_choice, computer_choice) in winning_pairs:
        return "Player Wins"
    return "Computer Wins"


class RockPaperScissorsGame:
    TARGET_SCORE = 3
    COUNTDOWN_SECONDS = 3.0
    RESULT_SECONDS = 2.5

    def __init__(self) -> None:
        self.player_score = 0
        self.computer_score = 0
        self.draw_score = 0
        self.state = GameState.WAITING
        self.player_choice = "--"
        self.computer_choice = "--"
        self.last_result = ""
        self.status_message = "Rounds start automatically."
        self._countdown_started_at = 0.0
        self._result_started_at = 0.0

    def start_round(self, current_time: float) -> None:
        if (
            self.state != GameState.WAITING
            or self.player_score >= self.TARGET_SCORE
            or self.computer_score >= self.TARGET_SCORE
        ):
            return

        self._begin_round(current_time)

    def _begin_round(self, current_time: float) -> None:
        self.state = GameState.COUNTDOWN
        self._countdown_started_at = current_time
        self.player_choice = "--"
        self.computer_choice = "--"
        self.last_result = ""
        self.status_message = "Get ready. Show one hand after the countdown."

    def countdown_number(self, current_time: float) -> int:
        if self.state != GameState.COUNTDOWN:
            return 0
        remaining = self.COUNTDOWN_SECONDS - (
            current_time - self._countdown_started_at
        )
        if remaining <= 0:
            return 0
        return ceil(remaining)

    def update(self, gesture: str, hands_detected: int, current_time: float) -> None:
        if self.state == GameState.COUNTDOWN:
            if current_time - self._countdown_started_at >= self.COUNTDOWN_SECONDS:
                self.state = GameState.PLAYING
                self.status_message = "Show one clear Rock, Paper, or Scissors gesture."
            return

        if self.state == GameState.PLAYING:
            if hands_detected == 0:
                self.status_message = "No hand detected. Show one hand to the camera."
                return
            if hands_detected > 1:
                self.status_message = "Multiple hands detected. Show only one hand."
                return
            if gesture not in CHOICES:
                self.status_message = "Gesture not recognized. Try a clear hand pose."
                return

            self._finish_round(gesture, current_time)
            return

        if (
            self.state == GameState.RESULT
            and current_time - self._result_started_at >= self.RESULT_SECONDS
        ):
            if (
                self.player_score >= self.TARGET_SCORE
                or self.computer_score >= self.TARGET_SCORE
            ):
                self.state = GameState.GAME_OVER
                self.status_message = (
                    "You are the winner!"
                    if self.player_score >= self.TARGET_SCORE
                    else "The computer is the winner!"
                )
            else:
                self._begin_round(current_time)

    def _finish_round(self, player_choice: str, current_time: float) -> None:
        computer_choice = random.choice(CHOICES)
        result = determine_winner(player_choice, computer_choice)

        self.player_choice = player_choice
        self.computer_choice = computer_choice
        self.last_result = result
        if result == "Player Wins":
            self.player_score += 1
        elif result == "Computer Wins":
            self.computer_score += 1
        else:
            self.draw_score += 1

        self.state = GameState.RESULT
        self._result_started_at = current_time
        self.status_message = "Round complete. The next round starts automatically."

    @property
    def match_winner(self) -> str | None:
        if self.player_score >= self.TARGET_SCORE:
            return "Player"
        if self.computer_score >= self.TARGET_SCORE:
            return "Computer"
        return None

    def reset_score(self) -> None:
        self.player_score = 0
        self.computer_score = 0
        self.draw_score = 0
        self.state = GameState.WAITING
        self.player_choice = "--"
        self.computer_choice = "--"
        self.last_result = ""
        self.status_message = "Score reset. Rounds start automatically."

    def quit(self) -> None:
        self.state = GameState.GAME_OVER
