"""Run the OpenCV webcam Rock Paper Scissors game."""

import sys
import time

import cv2
import numpy as np

from game import GameState, RockPaperScissorsGame
from gesture import (
    GestureStabilizer,
    detect_gesture,
    detect_navigation_gesture,
    hand_size_ratio,
)
from hand_detector import HandDetector, HandTrackingWorker
from navigation import GestureNavigator
from sound import (
    SoundManager,
    SoundEvent,
    SoundEventTracker,
    button_sound_event,
)
from ui import GameUI, HEIGHT, WIDTH

WINDOW_NAME = "HandPlay - Rock Paper Scissors"
DEBUG_MODE = False
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480


def advance_round_gesture(
    game: RockPaperScissorsGame,
    stabilizer: GestureStabilizer,
    gesture: str,
    hands_detected: int,
    current_time: float,
    pending_shot_choice: str | None,
) -> tuple[str | None, str, bool]:
    """Sample through countdown, then submit the SHOOT snapshot once."""
    was_counting_down = game.state == GameState.COUNTDOWN
    observed_gesture = gesture if hands_detected == 1 else "Unknown"
    stable_gesture = stabilizer.update(observed_gesture, current_time)

    if was_counting_down:
        game.update("Unknown", hands_detected, current_time)
        if game.state == GameState.PLAYING:
            pending_shot_choice = (
                stable_gesture if stable_gesture != "Unknown" else None
            )
        display_gesture = stable_gesture if hands_detected == 1 else "Unknown"
        return (
            pending_shot_choice,
            display_gesture,
            observed_gesture in {"Rock", "Paper", "Scissors"}
            and stable_gesture != observed_gesture,
        )

    captured_at_shoot = pending_shot_choice is not None
    if captured_at_shoot:
        stable_gesture = pending_shot_choice
        hands_detected = 1
    game.update(stable_gesture, hands_detected, current_time)
    return (
        None,
        stable_gesture,
        observed_gesture in {"Rock", "Paper", "Scissors"}
        and stable_gesture != observed_gesture,
    )


def main() -> int:
    game = RockPaperScissorsGame()
    ui = GameUI()
    sound = SoundManager()
    sound_events = SoundEventTracker()
    navigator = GestureNavigator()
    choice_stabilizer = GestureStabilizer()
    screen = "loading"
    camera: cv2.VideoCapture | None = None
    detector: HandDetector | None = None
    tracker: HandTrackingWorker | None = None
    camera_error = ""
    camera_frame: np.ndarray | None = None
    last_tracking_sequence = 0
    hands_detected = 0
    gesture = "Unknown"
    display_gesture = "Unknown"
    gesture_pending = False
    gesture_hint = "Show your hand"
    pending_shot_choice: str | None = None
    navigation_gesture = "Unknown"
    hand_cursor: tuple[int, int] | None = None
    previous_hand_detected = False
    debug_mode = DEBUG_MODE
    interaction_index_tip: tuple[int, int] | None = None
    go_feedback_until = 0.0
    detection_ms = 0.0
    tracking_fps = 0.0
    running = True

    def close_camera() -> None:
        nonlocal camera, detector, tracker, camera_frame, last_tracking_sequence
        if tracker is not None:
            try:
                tracker.close()
            except RuntimeError as exc:
                print(f"Hand tracking cleanup error: {exc}", file=sys.stderr)
            tracker = None
        if camera is not None:
            try:
                camera.release()
            except cv2.error as exc:
                print(f"Webcam cleanup error: {exc}", file=sys.stderr)
            camera = None
        if detector is not None:
            try:
                detector.close()
            except (RuntimeError, ValueError) as exc:
                print(f"Hand detector cleanup error: {exc}", file=sys.stderr)
            detector = None
        camera_frame = None
        last_tracking_sequence = 0

    def begin_camera_loading() -> None:
        nonlocal screen, camera_error, pending_shot_choice
        choice_stabilizer.reset()
        pending_shot_choice = None
        if camera is not None and detector is not None and tracker is not None:
            game.reset_score()
            screen = "game"
            navigator.reset()
            return
        camera_error = ""
        screen = "loading"

    def start_camera() -> None:
        nonlocal camera, detector, tracker, screen, camera_error
        try:
            camera = cv2.VideoCapture(0)
            if not camera.isOpened():
                camera.release()
                camera = None
                raise RuntimeError(
                    "Webcam 0 did not open. Check the connection and Windows camera privacy settings."
                )
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
            camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            detector = HandDetector()
            tracker = HandTrackingWorker(camera, detector)
            screen = "home"
            sound.play(SoundEvent.CAMERA_READY)
        except (AttributeError, OSError, RuntimeError, ValueError, cv2.error) as exc:
            close_camera()
            camera_error = str(exc)
            print(f"Camera startup error: {exc}", file=sys.stderr)
            screen = "error"

    def finish_game() -> None:
        nonlocal screen, pending_shot_choice
        sound.play_final_result(game.player_score, game.computer_score)
        game.quit()
        choice_stabilizer.reset()
        pending_shot_choice = None
        screen = "summary"
        navigator.reset()

    def play_game_events(current_time: float) -> None:
        nonlocal go_feedback_until
        events = sound_events.observe(
            game.state,
            game.countdown_number(current_time),
            hands_detected,
            gesture,
            game.last_result,
        )
        for event in events:
            sound.play(event)
            if event == SoundEvent.GO:
                go_feedback_until = current_time + 0.45

    def process_tracking_frame(
        detected: list[list[tuple[int, int]]],
        frame_size: tuple[int, int],
        current_time: float,
    ) -> str | None:
        nonlocal hands_detected, gesture, gesture_hint, previous_hand_detected
        nonlocal pending_shot_choice, display_gesture, gesture_pending
        nonlocal navigation_gesture, hand_cursor

        frame_width, frame_height = frame_size
        hands_detected = len(detected)
        gesture = "Unknown"
        if (
            screen == "game"
            and game.state in {GameState.COUNTDOWN, GameState.PLAYING}
            and pending_shot_choice is None
            and hands_detected == 1
        ):
            gesture = detect_gesture(detected[0])
        if screen == "game":
            gesture_hint = (
                "Show your hand"
                if hands_detected == 0
                else "Show only one hand"
                if hands_detected > 1
                else "Move closer"
                if hand_size_ratio(
                    detected[0],
                    (frame_width, frame_height),
                ) < 0.13
                else "Hand detected"
            )
        if (
            screen != "game"
            and hands_detected == 1
            and not previous_hand_detected
        ):
            sound.play(SoundEvent.HAND_CONFIRMED)
        previous_hand_detected = hands_detected == 1
        action: str | None = None

        if screen == "game":
            navigation_gesture = (
                detect_navigation_gesture(detected[0])
                if hands_detected == 1
                else "Unknown"
            )
            if game.state in {GameState.COUNTDOWN, GameState.PLAYING}:
                navigation_gesture = (
                    "Point" if navigation_gesture == "Point" else "Unknown"
                )
            nav_update = navigator.update(
                "game",
                navigation_gesture,
                detected[0][8] if hands_detected == 1 else None,
                frame_size,
                ui.action_at,
                current_time,
            )
            hand_cursor = nav_update.cursor
            hovered_action = (
                ui.action_at(hand_cursor)
                if hand_cursor is not None
                else None
            )
            if (
                nav_update.hover_changed
                and navigation_gesture == "Point"
                and not (hovered_action or "").startswith("volume_")
            ):
                sound.play(SoundEvent.BUTTON_HOVER)
            if nav_update.action is not None:
                action = nav_update.action
                sound.play(SoundEvent.BUTTON_CLICK)
            if game.state in {GameState.COUNTDOWN, GameState.PLAYING}:
                (
                    pending_shot_choice,
                    display_gesture,
                    gesture_pending,
                ) = advance_round_gesture(
                    game,
                    choice_stabilizer,
                    gesture,
                    hands_detected,
                    current_time,
                    pending_shot_choice,
                )
            else:
                choice_stabilizer.reset()
                display_gesture = gesture
                gesture_pending = False
                pending_shot_choice = None
            play_game_events(current_time)
        elif screen in {"home", "tutorial", "settings", "error", "summary"}:
            choice_stabilizer.reset()
            navigation_gesture = (
                detect_navigation_gesture(detected[0])
                if hands_detected == 1
                else "Unknown"
            )
            nav_update = navigator.update(
                screen,
                navigation_gesture,
                detected[0][8] if hands_detected == 1 else None,
                frame_size,
                ui.action_at,
                current_time,
            )
            hand_cursor = nav_update.cursor
            hovered_action = (
                ui.action_at(hand_cursor)
                if hand_cursor is not None
                else None
            )
            if (
                nav_update.hover_changed
                and navigation_gesture == "Point"
                and not (hovered_action or "").startswith("volume_")
            ):
                sound.play(SoundEvent.BUTTON_HOVER)
            if nav_update.action is not None:
                action = nav_update.action
                sound.play(
                    button_sound_event(action) or SoundEvent.BUTTON_CLICK
                )
        return action

    try:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, WIDTH, HEIGHT)
        cv2.setMouseCallback(WINDOW_NAME, ui.mouse_event)

        while running:
            action: str | None = None
            if screen == "loading":
                start_camera()

            snapshot = tracker.latest() if tracker is not None else None
            if tracker is not None:
                tracker.set_draw_landmarks(screen in {"home", "game"})
            if (
                screen in {"home", "tutorial", "settings", "error", "game", "summary"}
                and camera is not None
                and detector is not None
                and tracker is not None
                and snapshot is not None
                and (
                    snapshot.error is not None
                    or (
                        snapshot.frame is not None
                        and snapshot.sequence != last_tracking_sequence
                    )
                )
            ):
                now = time.monotonic()
                try:
                    if snapshot.error is not None:
                        raise RuntimeError(snapshot.error)
                    if snapshot.frame is None:
                        raise RuntimeError("The hand tracker produced no camera frame.")
                    last_tracking_sequence = snapshot.sequence
                    camera_frame = snapshot.frame
                    detection_ms = snapshot.detection_ms
                    tracking_fps = snapshot.fps
                    interaction_index_tip = (
                        snapshot.landmarks[0][8]
                        if len(snapshot.landmarks) == 1
                        else None
                    )
                    action = process_tracking_frame(
                        snapshot.landmarks,
                        (camera_frame.shape[1], camera_frame.shape[0]),
                        now,
                    )
                except (AttributeError, OSError, RuntimeError, ValueError, cv2.error) as exc:
                    camera_error = str(exc)
                    print(f"Camera or hand-detection error: {exc}", file=sys.stderr)
                    close_camera()
                    screen = "error"
                    navigator.reset()
                    choice_stabilizer.reset()
                    pending_shot_choice = None
                    hand_cursor = None
                    previous_hand_detected = False

            if screen == "game" and game.state == GameState.RESULT:
                game.update("Unknown", 0, time.monotonic())

            canvas = ui.render(
                screen,
                game,
                camera_frame,
                camera_active=camera is not None and camera.isOpened(),
                hands_detected=hands_detected,
                gesture=display_gesture,
                error_message=camera_error,
                sound_enabled=sound.enabled,
                volume_percent=sound.volume_percent,
                go_feedback=time.monotonic() < go_feedback_until,
                hand_position=hand_cursor,
                navigation_gesture=navigation_gesture,
                gesture_pending=gesture_pending,
                gesture_hint=gesture_hint,
            )
            if debug_mode:
                ui.draw_interaction_debug(
                    canvas,
                    hands_detected,
                    interaction_index_tip,
                    hand_cursor,
                    navigator.hovered_action,
                    navigator.interaction_state,
                    detection_ms,
                    tracking_fps,
                )
            _, _, viewport_width, viewport_height = cv2.getWindowImageRect(
                WINDOW_NAME
            )
            if viewport_width > 0 and viewport_height > 0:
                canvas = ui.fit_to_viewport(
                    canvas,
                    (viewport_width, viewport_height),
                )
            cv2.imshow(WINDOW_NAME, canvas)

            key = cv2.waitKey(16) & 0xFF
            if key == ord("d"):
                debug_mode = not debug_mode
            mouse_action = ui.take_action()
            if mouse_action is not None:
                action = mouse_action
            button_event = button_sound_event(mouse_action)
            if button_event is not None:
                sound.play(button_event)

            if key in (ord("q"), 27) and screen == "game":
                finish_game()
            elif key == ord("q") and screen in (
                "home",
                "tutorial",
                "settings",
                "error",
                "summary",
            ):
                running = False
            elif screen == "home":
                if action == "exit":
                    running = False
                elif action == "start" or key in (13, 32):
                    begin_camera_loading()
                elif action == "tutorial" or key == ord("h"):
                    screen = "tutorial"
                    navigator.reset()
                elif action == "settings":
                    screen = "settings"
                    navigator.reset()
            elif screen == "tutorial":
                if action in {"home", "back"} or key in (27, 8):
                    screen = "home"
                    navigator.reset()
                elif action == "start" or key in (13, 32):
                    begin_camera_loading()
            elif screen == "settings":
                if action == "toggle_sound" or key in (13, 32):
                    sound.enabled = not sound.enabled
                elif action is not None and action.startswith("volume_"):
                    sound.set_volume(int(action.removeprefix("volume_")))
                elif action in {"home", "back"} or key in (27, 8):
                    screen = "home"
                    navigator.reset()
            elif screen == "loading":
                if key == ord("q"):
                    running = False
            elif screen == "error":
                if action == "retry" or key in (13, ord("r")):
                    begin_camera_loading()
                elif action in {"home", "back"} or key in (27, 8):
                    screen = "home"
                    navigator.reset()
            elif screen == "game":
                if action in {"finish", "back"} or key == ord("q"):
                    finish_game()
                elif action == "next_round" and game.state == GameState.WAITING:
                    game.start_round(time.monotonic())
                elif key == ord(" ") and game.state == GameState.WAITING:
                    game.start_round(time.monotonic())
                elif key == ord("r"):
                    game.reset_score()
            elif screen == "summary":
                if action == "exit":
                    running = False
                elif action == "play_again" or key in (ord("p"), 13, 32):
                    game.reset_score()
                    screen = "game"
                    navigator.reset()
                elif action in {"home", "back"} or key in (ord("h"), 27, 8):
                    game.reset_score()
                    screen = "home"
                    navigator.reset()
                elif key == ord("q"):
                    running = False

            try:
                if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    running = False
            except cv2.error:
                running = False

    except (AttributeError, OSError, RuntimeError, ValueError, cv2.error) as exc:
        print(f"Game UI error: {exc}", file=sys.stderr)
        return 1
    finally:
        try:
            sound.close()
        except (OSError, RuntimeError) as exc:
            print(f"Sound cleanup error: {exc}", file=sys.stderr)
        close_camera()
        try:
            cv2.destroyAllWindows()
        except cv2.error as exc:
            print(f"Window cleanup error: {exc}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
