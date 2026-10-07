import unittest
from pathlib import Path
import wave
from unittest.mock import Mock, patch

import winsound

from game import GameState
from sound import (
    SOUND_FILES,
    SoundManager,
    SoundEvent,
    SoundEventTracker,
    button_sound_event,
    final_sound_event,
)


class SoundEventTrackerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tracker = SoundEventTracker()

    def observe(
        self,
        state: GameState,
        countdown: int = 0,
        hands: int = 0,
        gesture: str = "Unknown",
        result: str = "",
    ) -> tuple[SoundEvent, ...]:
        return self.tracker.observe(state, countdown, hands, gesture, result)

    def test_countdown_and_go_events_fire_once_per_transition(self) -> None:
        self.assertEqual(
            self.observe(GameState.COUNTDOWN, countdown=3),
            (SoundEvent.COUNTDOWN,),
        )
        self.assertEqual(self.observe(GameState.COUNTDOWN, countdown=3), ())
        self.assertEqual(
            self.observe(GameState.COUNTDOWN, countdown=2),
            (SoundEvent.COUNTDOWN,),
        )
        self.assertEqual(
            self.observe(GameState.COUNTDOWN, countdown=1),
            (SoundEvent.COUNTDOWN_FINAL,),
        )
        self.assertEqual(self.observe(GameState.COUNTDOWN, countdown=1), ())
        self.assertEqual(
            self.observe(GameState.PLAYING),
            (SoundEvent.GO,),
        )
        self.assertEqual(self.observe(GameState.PLAYING), ())

    def test_shoot_transition_does_not_stack_hand_confirmation_sound(self) -> None:
        self.observe(GameState.COUNTDOWN, countdown=1, hands=1)
        self.assertEqual(
            self.observe(GameState.PLAYING, hands=1, gesture="Rock"),
            (SoundEvent.GO,),
        )

    def test_hand_confirmation_does_not_repeat_while_hand_remains_visible(self) -> None:
        self.assertEqual(
            self.observe(GameState.PLAYING, hands=1, gesture="Rock"),
            (SoundEvent.HAND_CONFIRMED,),
        )
        self.assertEqual(
            self.observe(GameState.PLAYING, hands=1),
            (),
        )
        self.observe(GameState.PLAYING, hands=0)
        self.assertEqual(
            self.observe(GameState.PLAYING, hands=1),
            (SoundEvent.HAND_CONFIRMED,),
        )

    def test_round_result_event_fires_once_for_each_result(self) -> None:
        result_events = {
            "Player Wins": SoundEvent.ROUND_WIN,
            "Computer Wins": SoundEvent.ROUND_LOSE,
            "Draw": SoundEvent.DRAW,
        }
        for result, expected in result_events.items():
            with self.subTest(result=result):
                tracker = SoundEventTracker()
                tracker.observe(GameState.PLAYING, 0, 0, "Unknown", "")
                self.assertEqual(
                    tracker.observe(GameState.RESULT, 0, 1, "Rock", result),
                    (expected,),
                )
                self.assertEqual(
                    tracker.observe(GameState.RESULT, 0, 1, "Rock", result),
                    (),
                )

    def test_button_and_final_sound_mappings(self) -> None:
        for action in (
            "start",
            "tutorial",
            "next_round",
            "play_again",
            "home",
            "retry",
            "exit",
            "back",
            "toggle_sound",
        ):
            with self.subTest(action=action):
                self.assertEqual(button_sound_event(action), SoundEvent.BUTTON_CLICK)
        self.assertEqual(button_sound_event("volume_50"), SoundEvent.BUTTON_CLICK)
        self.assertIsNone(button_sound_event("finish_round"))
        self.assertEqual(final_sound_event(2, 1), SoundEvent.FINAL_WIN)
        self.assertEqual(final_sound_event(1, 2), SoundEvent.FINAL_LOSE)
        self.assertEqual(final_sound_event(1, 1), SoundEvent.FINAL_DRAW)


class SoundManagerTests(unittest.TestCase):
    def test_all_bundled_effects_are_short_and_at_comfortable_levels(self) -> None:
        sounds_directory = Path(__file__).resolve().parents[1] / "assets" / "sounds"
        self.assertEqual(len(SOUND_FILES), 13)
        for filename in SOUND_FILES.values():
            with self.subTest(filename=filename):
                with wave.open(str(sounds_directory / filename), "rb") as sound_file:
                    duration = sound_file.getnframes() / sound_file.getframerate()
                    samples = sound_file.readframes(sound_file.getnframes())
                self.assertLessEqual(duration, 2.0)
                peak = max(
                    abs(int.from_bytes(samples[index:index + 2], "little", signed=True))
                    for index in range(0, len(samples), 2)
                )
                self.assertLessEqual(peak, 11000)

    @patch("sound.winsound.PlaySound")
    def test_play_is_async_and_sound_can_be_disabled(self, play_sound: Mock) -> None:
        sound = SoundManager()
        click_path = (
            Path(__file__).resolve().parents[1]
            / "assets"
            / "sounds"
            / SOUND_FILES[SoundEvent.BUTTON_CLICK]
        )
        sound.play(SoundEvent.BUTTON_CLICK)
        play_sound.assert_called_once_with(
            unittest.mock.ANY,
            winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_NODEFAULT,
        )
        self.assertTrue(Path(play_sound.call_args.args[0]).is_file())
        self.assertNotEqual(Path(play_sound.call_args.args[0]), click_path)

        play_sound.reset_mock()
        sound.enabled = False
        sound.play(SoundEvent.ROUND_WIN)
        play_sound.assert_not_called()
        sound.close()

    @patch("sound.winsound.PlaySound")
    def test_volume_changes_scale_cached_playback_assets(
        self, play_sound: Mock
    ) -> None:
        sound = SoundManager(volume_percent=40)
        sound.play(SoundEvent.ROUND_WIN)
        quiet_path = Path(play_sound.call_args.args[0])

        sound.set_volume(80)
        sound.play(SoundEvent.ROUND_WIN)
        loud_path = Path(play_sound.call_args.args[0])

        self.assertNotEqual(quiet_path, loud_path)
        with wave.open(str(quiet_path), "rb") as quiet, wave.open(
            str(loud_path), "rb"
        ) as loud:
            quiet_data = quiet.readframes(quiet.getnframes())
            loud_data = loud.readframes(loud.getnframes())
        quiet_peak = max(
            abs(int.from_bytes(quiet_data[index:index + 2], "little", signed=True))
            for index in range(0, len(quiet_data), 2)
        )
        loud_peak = max(
            abs(int.from_bytes(loud_data[index:index + 2], "little", signed=True))
            for index in range(0, len(loud_data), 2)
        )
        self.assertGreater(loud_peak, quiet_peak)

        with self.assertRaises(ValueError):
            sound.set_volume(101)
        sound.close()


if __name__ == "__main__":
    unittest.main()
