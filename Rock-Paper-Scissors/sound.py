"""Event-driven sound effects for the Windows OpenCV game."""

from array import array
from enum import Enum, auto
from pathlib import Path
import sys
import tempfile
import wave

import winsound

from game import GameState


class SoundEvent(Enum):
    BUTTON_CLICK = auto()
    BUTTON_HOVER = auto()
    CAMERA_READY = auto()
    HAND_CONFIRMED = auto()
    COUNTDOWN = auto()
    COUNTDOWN_FINAL = auto()
    GO = auto()
    ROUND_WIN = auto()
    ROUND_LOSE = auto()
    DRAW = auto()
    FINAL_WIN = auto()
    FINAL_LOSE = auto()
    FINAL_DRAW = auto()


SOUND_FILES = {
    SoundEvent.BUTTON_CLICK: "click.wav",
    SoundEvent.BUTTON_HOVER: "hover.wav",
    SoundEvent.CAMERA_READY: "camera_ready.wav",
    SoundEvent.HAND_CONFIRMED: "hand_confirmed.wav",
    SoundEvent.COUNTDOWN: "countdown.wav",
    SoundEvent.COUNTDOWN_FINAL: "countdown_final.wav",
    SoundEvent.GO: "go.wav",
    SoundEvent.ROUND_WIN: "win.wav",
    SoundEvent.ROUND_LOSE: "lose.wav",
    SoundEvent.DRAW: "draw.wav",
    SoundEvent.FINAL_WIN: "final_win.wav",
    SoundEvent.FINAL_LOSE: "final_lose.wav",
    SoundEvent.FINAL_DRAW: "final_draw.wav",
}


def final_sound_event(player_score: int, computer_score: int) -> SoundEvent:
    if player_score > computer_score:
        return SoundEvent.FINAL_WIN
    if player_score < computer_score:
        return SoundEvent.FINAL_LOSE
    return SoundEvent.FINAL_DRAW


def button_sound_event(action: str | None) -> SoundEvent | None:
    if action in {
        "start",
        "tutorial",
        "next_round",
        "play_again",
        "home",
        "retry",
        "exit",
        "finish",
        "back",
        "toggle_sound",
    } or (action is not None and action.startswith("volume_")):
        return SoundEvent.BUTTON_CLICK
    return None


class SoundManager:
    """Play bundled WAV effects asynchronously at a user-selected master level."""

    def __init__(
        self,
        sounds_directory: Path | None = None,
        enabled: bool = True,
        volume_percent: int = 80,
    ) -> None:
        self.enabled = enabled
        self.volume_percent = 80
        self.set_volume(volume_percent)
        self._sounds_directory = (
            sounds_directory
            if sounds_directory is not None
            else Path(__file__).resolve().parent / "assets" / "sounds"
        )
        self._reported_errors: set[SoundEvent] = set()
        self._temporary_directory = tempfile.TemporaryDirectory(
            prefix="handplay-sounds-"
        )
        self._scaled_paths: dict[tuple[SoundEvent, int], Path] = {}

    def set_volume(self, volume_percent: int) -> None:
        if not 0 <= volume_percent <= 100:
            raise ValueError("Sound volume must be between 0 and 100.")
        self.volume_percent = volume_percent

    def play(self, event: SoundEvent) -> None:
        if not self.enabled or self.volume_percent == 0:
            return

        sound_path = self._sounds_directory / SOUND_FILES[event]
        if not sound_path.is_file():
            if event not in self._reported_errors:
                print(f"Sound file not found: {sound_path}", file=sys.stderr)
                self._reported_errors.add(event)
            return

        try:
            playback_path = self._playback_path(event, sound_path)
            winsound.PlaySound(
                str(playback_path),
                winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_NODEFAULT,
            )
        except (OSError, RuntimeError, ValueError, wave.Error) as exc:
            if event not in self._reported_errors:
                print(f"Could not play {sound_path}: {exc}", file=sys.stderr)
                self._reported_errors.add(event)

    def play_final_result(self, player_score: int, computer_score: int) -> None:
        self.play(final_sound_event(player_score, computer_score))

    def _playback_path(self, event: SoundEvent, source_path: Path) -> Path:
        if self.volume_percent == 100:
            return source_path

        cache_key = (event, self.volume_percent)
        cached_path = self._scaled_paths.get(cache_key)
        if cached_path is not None:
            return cached_path

        with wave.open(str(source_path), "rb") as source:
            if source.getsampwidth() != 2:
                raise wave.Error("Sound effects must use 16-bit PCM.")
            parameters = source.getparams()
            samples = array("h")
            samples.frombytes(source.readframes(source.getnframes()))
        if sys.byteorder != "little":
            samples.byteswap()
        scale = self.volume_percent / 100
        for index, sample in enumerate(samples):
            samples[index] = round(sample * scale)
        if sys.byteorder != "little":
            samples.byteswap()

        target_path = (
            Path(self._temporary_directory.name)
            / f"{event.name.lower()}_{self.volume_percent}.wav"
        )
        with wave.open(str(target_path), "wb") as target:
            target.setparams(parameters)
            target.writeframes(samples.tobytes())
        self._scaled_paths[cache_key] = target_path
        return target_path

    def close(self) -> None:
        try:
            winsound.PlaySound(None, 0)
        finally:
            self._temporary_directory.cleanup()
            self._scaled_paths.clear()


class SoundEventTracker:
    """Turn per-frame game state into one-shot audio events."""

    def __init__(self) -> None:
        self._previous_state: GameState | None = None
        self._last_countdown_number: int | None = None
        self._hand_detected = False

    def observe(
        self,
        state: GameState,
        countdown_number: int,
        hands_detected: int,
        gesture: str,
        last_result: str,
    ) -> tuple[SoundEvent, ...]:
        events: list[SoundEvent] = []

        if state == GameState.COUNTDOWN:
            if self._previous_state != GameState.COUNTDOWN:
                self._last_countdown_number = None
            if (
                countdown_number > 0
                and countdown_number != self._last_countdown_number
            ):
                events.append(
                    SoundEvent.COUNTDOWN_FINAL
                    if countdown_number == 1
                    else SoundEvent.COUNTDOWN
                )
                self._last_countdown_number = countdown_number
        else:
            self._last_countdown_number = None

        if (
            self._previous_state == GameState.COUNTDOWN
            and state == GameState.PLAYING
        ):
            events.append(SoundEvent.GO)

        if (
            self._previous_state == GameState.PLAYING
            and state == GameState.RESULT
        ):
            result_events = {
                "Player Wins": SoundEvent.ROUND_WIN,
                "Computer Wins": SoundEvent.ROUND_LOSE,
                "Draw": SoundEvent.DRAW,
            }
            result_event = result_events.get(last_result)
            if result_event is not None:
                events.append(result_event)

        if state == GameState.PLAYING:
            hand_detected = hands_detected == 1
            if (
                self._previous_state != GameState.COUNTDOWN
                and hand_detected
                and not self._hand_detected
            ):
                events.append(SoundEvent.HAND_CONFIRMED)
            self._hand_detected = hand_detected
        else:
            self._hand_detected = False

        self._previous_state = state
        return tuple(events)
