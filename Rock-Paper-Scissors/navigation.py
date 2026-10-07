"""Hand-pointer dwell selection and gesture navigation controls."""

from dataclasses import dataclass
import time
from collections.abc import Callable

from ui import HEIGHT, WIDTH


@dataclass(frozen=True)
class NavigationUpdate:
    cursor: tuple[int, int] | None
    action: str | None = None
    hover_changed: bool = False


class GestureNavigator:
    RELEASE_SECONDS = 0.12
    BACK_SECONDS = 0.65
    COOLDOWN_SECONDS = 0.25
    CURSOR_TIMEOUT_SECONDS = 0.15
    CURSOR_DEAD_ZONE = 2
    SMOOTHING = 0.9
    CLICK_CONFIRM_FRAMES = 3
    CLICK_MAX_DRIFT = 24

    def __init__(self) -> None:
        self._cursor: tuple[int, int] | None = None
        self._last_point_time = 0.0
        self._hover_action: str | None = None
        self._selection_action: str | None = None
        self._selection_frames = 0
        self._selection_origin: tuple[int, int] | None = None
        self._gesture: str | None = None
        self._gesture_started_at = 0.0
        self._cooldown_until = 0.0
        self._point_released = True
        self._release_started_at: float | None = None
        self._screen: str | None = None

    @property
    def cursor(self) -> tuple[int, int] | None:
        return self._cursor

    @property
    def hovered_action(self) -> str | None:
        return self._hover_action

    @property
    def interaction_state(self) -> str:
        if not self._point_released:
            return "WAITING"
        if self._selection_action is not None and self._selection_frames:
            return "PRESSED"
        return "READY"

    def update(
        self,
        screen: str,
        gesture: str,
        index_tip: tuple[int, int] | None,
        frame_size: tuple[int, int],
        action_at: Callable[[tuple[int, int]], str | None],
        current_time: float | None = None,
    ) -> NavigationUpdate:
        now = time.monotonic() if current_time is None else current_time
        if screen != self._screen:
            self._reset_screen_state(screen)

        if index_tip is not None:
            frame_width, frame_height = frame_size
            target = (
                min(WIDTH - 1, max(0, round(index_tip[0] * WIDTH / frame_width))),
                min(HEIGHT - 1, max(0, round(index_tip[1] * HEIGHT / frame_height))),
            )
            if self._cursor is None:
                self._cursor = target
            else:
                delta_x = target[0] - self._cursor[0]
                delta_y = target[1] - self._cursor[1]
                if (
                    delta_x * delta_x + delta_y * delta_y
                    <= self.CURSOR_DEAD_ZONE**2
                ):
                    target = self._cursor
                self._cursor = (
                    round(self._cursor[0] * (1 - self.SMOOTHING) + target[0] * self.SMOOTHING),
                    round(self._cursor[1] * (1 - self.SMOOTHING) + target[1] * self.SMOOTHING),
                )
            self._last_point_time = now

        visible_cursor = self._visible_cursor(now)
        cursor_is_fresh = now - self._last_point_time <= self.CURSOR_TIMEOUT_SECONDS
        if cursor_is_fresh and visible_cursor is not None:
            hover_action = action_at(visible_cursor)
        else:
            hover_action = None
        hover_changed = hover_action != self._hover_action
        if hover_changed:
            self._hover_action = hover_action

        if not cursor_is_fresh:
            self._cursor = None
            self._selection_action = None
            self._selection_frames = 0
            self._selection_origin = None

        selection_action = (
            hover_action
            if gesture == "Point" and hover_action is not None
            else None
        )
        if selection_action != self._selection_action:
            self._selection_action = selection_action
            self._selection_frames = 1 if selection_action is not None else 0
            self._selection_origin = (
                visible_cursor if selection_action is not None else None
            )
        elif selection_action is not None and visible_cursor is not None:
            origin = self._selection_origin or visible_cursor
            drift_x = visible_cursor[0] - origin[0]
            drift_y = visible_cursor[1] - origin[1]
            if drift_x * drift_x + drift_y * drift_y > self.CLICK_MAX_DRIFT**2:
                self._selection_frames = 1
                self._selection_origin = visible_cursor
            else:
                self._selection_frames += 1

        if gesture != "Point":
            if self._release_started_at is None:
                self._release_started_at = now
            elif (
                now - self._release_started_at
                >= self.RELEASE_SECONDS
            ):
                self._point_released = True
        else:
            self._release_started_at = None

        if now < self._cooldown_until:
            return NavigationUpdate(self._visible_cursor(now), None, hover_changed)

        if screen in {"tutorial", "settings", "error", "summary", "game"}:
            if gesture == "Fist":
                action = self._hold_action(gesture, now, self.BACK_SECONDS)
                if action:
                    self._cooldown_until = now + self.COOLDOWN_SECONDS
                    return NavigationUpdate(self._visible_cursor(now), "back", hover_changed)
            else:
                self._clear_gesture_hold()

        if (
            selection_action is not None
            and gesture == "Point"
            and self._point_released
            and self._selection_frames >= self.CLICK_CONFIRM_FRAMES
        ):
            self._cooldown_until = now + self.COOLDOWN_SECONDS
            self._point_released = False
            self._release_started_at = None
            self._selection_frames = 0
            self._selection_origin = None
            return NavigationUpdate(
                self._visible_cursor(now),
                hover_action,
                hover_changed,
            )
        if gesture != "Fist":
            self._clear_gesture_hold()

        return NavigationUpdate(self._visible_cursor(now), None, hover_changed)

    def reset(self) -> None:
        self._reset_screen_state(None)
        self._cursor = None
        self._last_point_time = 0.0

    def _hold_action(self, gesture: str, now: float, duration: float) -> bool:
        if gesture != self._gesture:
            self._gesture = gesture
            self._gesture_started_at = now
            return False
        return now - self._gesture_started_at >= duration

    def _clear_gesture_hold(self) -> None:
        self._gesture = None
        self._gesture_started_at = 0.0

    def _visible_cursor(self, now: float) -> tuple[int, int] | None:
        if now - self._last_point_time > self.CURSOR_TIMEOUT_SECONDS:
            return None
        return self._cursor

    def _reset_screen_state(self, screen: str | None) -> None:
        self._screen = screen
        self._cursor = None
        self._last_point_time = 0.0
        self._hover_action = None
        self._selection_action = None
        self._selection_frames = 0
        self._selection_origin = None
        self._clear_gesture_hold()
