"""OpenCV-rendered screens and controls for the hand-gesture game."""

from functools import lru_cache
import os
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from game import GameState, RockPaperScissorsGame

WIDTH = 1360
HEIGHT = 860

BG = (42, 23, 15)
BG_LIGHT = (59, 41, 30)
CARD = (62, 46, 35)
CARD_LIGHT = (79, 59, 47)
ACCENT = (248, 189, 56)
ACCENT_DARK = (183, 119, 28)
BUTTON_PRIMARY = (255, 220, 0)
BUTTON_PRIMARY_HOVER = (255, 245, 72)
SUCCESS = (94, 197, 34)
DANGER = (68, 68, 239)
WARNING = (11, 158, 245)
COMPUTER = (246, 143, 167)
TEXT = (252, 250, 248)
MUTED = (177, 163, 148)
LINE = (93, 74, 58)
SHADOW = (24, 16, 11)


@lru_cache(maxsize=96)
def _font(size: int, bold: bool) -> ImageFont.FreeTypeFont:
    fonts = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
    preferred = fonts / ("segoeuib.ttf" if bold else "segoeui.ttf")
    fallback = fonts / ("arialbd.ttf" if bold else "arial.ttf")
    if preferred.is_file():
        return ImageFont.truetype(str(preferred), size)
    if fallback.is_file():
        return ImageFont.truetype(str(fallback), size)
    return ImageFont.truetype("DejaVuSans.ttf", size)


@lru_cache(maxsize=512)
def _text_mask(value: str, size: int, bold: bool) -> tuple[np.ndarray, int, int, int, int]:
    font = _font(size, bold)
    left, top, right, bottom = font.getbbox(value, anchor="ls")
    padding = 2
    mask = Image.new("L", (right - left + padding * 2, bottom - top + padding * 2))
    ImageDraw.Draw(mask).text(
        (padding - left, padding - top),
        value,
        font=font,
        fill=255,
        anchor="ls",
    )
    mask_array = np.asarray(mask, dtype=np.float32) / 255.0
    mask_array.setflags(write=False)
    return mask_array, left, top, right, bottom


def _draw_label(
    image: np.ndarray,
    value: str,
    baseline: tuple[int, int],
    size: float,
    color: tuple[int, int, int],
    thickness: int,
    centered: bool,
) -> None:
    pixel_size = max(16, round(size * 40))
    mask_array, left, top, right, bottom = _text_mask(
        value,
        pixel_size,
        thickness >= 2,
    )
    if centered:
        origin_x = round(baseline[0] - (left + right) / 2)
        origin_y = round(baseline[1] - (top + bottom) / 2)
    else:
        origin_x, origin_y = baseline

    padding = 2
    x1 = max(0, origin_x + left - padding)
    y1 = max(0, origin_y + top - padding)
    x2 = min(image.shape[1], origin_x + right + padding)
    y2 = min(image.shape[0], origin_y + bottom + padding)
    if x1 >= x2 or y1 >= y2:
        return
    cropped_mask = mask_array[
        y1 - (origin_y + top - padding):y2 - (origin_y + top - padding),
        x1 - (origin_x + left - padding):x2 - (origin_x + left - padding),
        None,
    ]
    region = image[y1:y2, x1:x2]
    foreground = np.asarray(color, dtype=np.float32)
    region[:] = np.clip(
        region.astype(np.float32) * (1.0 - cropped_mask)
        + foreground * cropped_mask,
        0,
        255,
    ).astype(np.uint8)


def _rounded_rect(
    image: np.ndarray,
    box: tuple[int, int, int, int],
    color: tuple[int, int, int],
    radius: int = 18,
    thickness: int = -1,
) -> None:
    x1, y1, x2, y2 = box
    radius = min(radius, max(0, (x2 - x1) // 2), max(0, (y2 - y1) // 2))
    if thickness < 0:
        cv2.rectangle(image, (x1 + radius, y1), (x2 - radius, y2), color, -1)
        cv2.rectangle(image, (x1, y1 + radius), (x2, y2 - radius), color, -1)
        for point in (
            (x1 + radius, y1 + radius),
            (x2 - radius, y1 + radius),
            (x1 + radius, y2 - radius),
            (x2 - radius, y2 - radius),
        ):
            cv2.circle(image, point, radius, color, -1, cv2.LINE_AA)
        return

    cv2.line(image, (x1 + radius, y1), (x2 - radius, y1), color, thickness, cv2.LINE_AA)
    cv2.line(image, (x1 + radius, y2), (x2 - radius, y2), color, thickness, cv2.LINE_AA)
    cv2.line(image, (x1, y1 + radius), (x1, y2 - radius), color, thickness, cv2.LINE_AA)
    cv2.line(image, (x2, y1 + radius), (x2, y2 - radius), color, thickness, cv2.LINE_AA)
    for point in (
        (x1 + radius, y1 + radius),
        (x2 - radius, y1 + radius),
        (x1 + radius, y2 - radius),
        (x2 - radius, y2 - radius),
    ):
        cv2.circle(image, point, radius, color, thickness, cv2.LINE_AA)


def _text(
    image: np.ndarray,
    value: str,
    origin: tuple[int, int],
    size: float = 0.6,
    color: tuple[int, int, int] = TEXT,
    thickness: int = 1,
) -> None:
    _draw_label(image, value, origin, size, color, thickness, centered=False)


def _center_text(
    image: np.ndarray,
    value: str,
    center: tuple[int, int],
    size: float = 0.6,
    color: tuple[int, int, int] = TEXT,
    thickness: int = 1,
) -> None:
    _draw_label(
        image,
        value,
        center,
        size,
        color,
        thickness,
        centered=True,
    )


def _panel(
    image: np.ndarray,
    box: tuple[int, int, int, int],
    color: tuple[int, int, int] = CARD,
    radius: int = 18,
) -> None:
    x1, y1, x2, y2 = box
    _rounded_rect(image, (x1, y1 + 6, x2, y2 + 6), SHADOW, radius)
    _rounded_rect(image, box, color, radius)


def _draw_gesture_icon(
    image: np.ndarray,
    center: tuple[int, int],
    gesture: str,
    color: tuple[int, int, int] = ACCENT,
) -> None:
    x, y = center
    if gesture == "rock":
        _rounded_rect(image, (x - 14, y - 7, x + 14, y + 19), color, 7, 2)
        cv2.line(image, (x - 10, y - 2), (x + 10, y - 2), color, 3, cv2.LINE_AA)
        cv2.line(image, (x - 14, y + 7), (x - 21, y + 1), color, 5, cv2.LINE_AA)
    else:
        _rounded_rect(image, (x - 13, y + 2, x + 13, y + 22), color, 7, 2)
        if gesture == "paper":
            for offset, top in ((-10, y - 16), (-3, y - 23), (4, y - 24), (11, y - 15)):
                cv2.line(image, (x + offset, y + 5), (x + offset, top), color, 5, cv2.LINE_AA)
            cv2.line(image, (x - 12, y + 10), (x - 22, y + 1), color, 5, cv2.LINE_AA)
        else:
            cv2.line(image, (x - 7, y + 5), (x - 15, y - 19), color, 5, cv2.LINE_AA)
            cv2.line(image, (x + 7, y + 5), (x + 15, y - 19), color, 5, cv2.LINE_AA)
            cv2.line(image, (x - 12, y + 8), (x + 10, y + 8), color, 4, cv2.LINE_AA)


class GameUI:
    """Draw polished application screens and translate clicks to UI actions."""

    BUTTON_HIT_PADDING = 8

    def __init__(self) -> None:
        self._buttons: list[tuple[tuple[int, int, int, int], str]] = []
        self._click: tuple[int, int] | None = None
        self._mouse_position = (-1, -1)
        self._hand_position: tuple[int, int] | None = None
        self._sliders: list[tuple[int, int, int, int]] = []
        self._last_screen: str | None = None
        self._transition_canvas: np.ndarray | None = None
        self._transition_started_at = 0.0
        self._last_game_state: GameState | None = None
        self._result_animation_started_at = 0.0
        self._viewport_scale = 1.0
        self._viewport_offset = (0, 0)
        self._button_hover: dict[str, float] = {}
        self._last_render_time: float | None = None
        self._render_delta = 1 / 60

    def mouse_event(self, event: int, x: int, y: int, _flags: int, _param: object) -> None:
        offset_x, offset_y = self._viewport_offset
        logical_x = round((x - offset_x) / self._viewport_scale)
        logical_y = round((y - offset_y) / self._viewport_scale)
        inside_canvas = 0 <= logical_x < WIDTH and 0 <= logical_y < HEIGHT
        self._mouse_position = (
            (logical_x, logical_y) if inside_canvas else (-1, -1)
        )
        if event == cv2.EVENT_LBUTTONDOWN:
            self._click = self._mouse_position if inside_canvas else None

    def fit_to_viewport(
        self,
        canvas: np.ndarray,
        viewport_size: tuple[int, int],
    ) -> np.ndarray:
        """Scale the fixed logical layout into a viewport without changing geometry."""
        viewport_width, viewport_height = viewport_size
        if viewport_width <= 0 or viewport_height <= 0:
            raise ValueError("Viewport dimensions must be positive.")

        scale = min(viewport_width / WIDTH, viewport_height / HEIGHT)
        content_width = max(1, round(WIDTH * scale))
        content_height = max(1, round(HEIGHT * scale))
        offset_x = (viewport_width - content_width) // 2
        offset_y = (viewport_height - content_height) // 2
        self._viewport_scale = scale
        self._viewport_offset = (offset_x, offset_y)

        if (content_width, content_height) == (viewport_width, viewport_height):
            if canvas.shape[:2] == (viewport_height, viewport_width):
                return canvas
            return cv2.resize(
                canvas,
                (viewport_width, viewport_height),
                interpolation=(
                    cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
                ),
            )

        frame = np.full(
            (viewport_height, viewport_width, 3),
            BG,
            dtype=np.uint8,
        )
        resized = cv2.resize(
            canvas,
            (content_width, content_height),
            interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
        )
        frame[
            offset_y:offset_y + content_height,
            offset_x:offset_x + content_width,
        ] = resized
        return frame

    def take_action(self) -> str | None:
        if self._click is None:
            return None
        x, y = self._click
        self._click = None
        return self.action_at((x, y))

    def action_at(self, position: tuple[int, int]) -> str | None:
        x, y = position
        for x1, y1, x2, y2 in self._sliders:
            if x1 <= x <= x2 and y1 <= y <= y2:
                value = round((x - x1) / (x2 - x1) * 10) * 10
                return f"volume_{value}"
        for (x1, y1, x2, y2), action in reversed(self._buttons):
            if x1 <= x <= x2 and y1 <= y <= y2:
                return action
        return None

    def render(
        self,
        screen: str,
        game: RockPaperScissorsGame,
        video_frame: np.ndarray | None = None,
        camera_active: bool = False,
        hands_detected: int = 0,
        gesture: str = "Unknown",
        error_message: str = "",
        sound_enabled: bool = True,
        volume_percent: int = 80,
        go_feedback: bool = False,
        hand_position: tuple[int, int] | None = None,
        navigation_gesture: str = "Unknown",
        gesture_pending: bool = False,
        gesture_hint: str = "Show your hand",
    ) -> np.ndarray:
        render_time = time.monotonic()
        if self._last_render_time is not None:
            self._render_delta = min(
                0.1,
                max(0.0, render_time - self._last_render_time),
            )
        self._last_render_time = render_time
        canvas = self._background()
        self._buttons = []
        self._sliders = []
        self._hand_position = hand_position
        result_animation_progress = 1.0
        if screen == "game":
            now = time.monotonic()
            if game.state != self._last_game_state:
                if game.state == GameState.RESULT:
                    self._result_animation_started_at = now
                self._last_game_state = game.state
            if game.state == GameState.RESULT:
                result_animation_progress = min(
                    1.0,
                    (now - self._result_animation_started_at) / 0.22,
                )
        if screen == "home":
            self._draw_home(canvas, video_frame)
        elif screen == "tutorial":
            self._draw_tutorial(canvas)
        elif screen == "settings":
            self._draw_settings(canvas, sound_enabled, volume_percent)
        elif screen == "loading":
            self._draw_message(canvas, "Starting camera...", "Preparing your camera and hand tracking.", "loading")
        elif screen == "error":
            self._draw_error(canvas, error_message)
        elif screen == "game":
            self._draw_game(
                canvas,
                game,
                video_frame,
                camera_active,
                hands_detected,
                gesture,
                go_feedback,
                result_animation_progress,
                gesture_pending,
                gesture_hint,
            )
        elif screen == "summary":
            self._draw_summary(canvas, game)
        if screen not in {"loading", "error"}:
            self._draw_hand_status(canvas, hands_detected, navigation_gesture)
        self._draw_hand_pointer(canvas, self._hand_position)
        if screen != self._last_screen:
            self._transition_canvas = (
                self._background() if self._last_screen is not None else None
            )
            self._transition_started_at = time.monotonic()
            self._last_screen = screen
        if self._transition_canvas is not None:
            progress = min(1.0, (time.monotonic() - self._transition_started_at) / 0.2)
            canvas = cv2.addWeighted(
                self._transition_canvas,
                1.0 - progress,
                canvas,
                progress,
                0,
            )
            if progress >= 1.0:
                self._transition_canvas = None
        return canvas

    def _background(self) -> np.ndarray:
        top = np.array((BG[0], BG[1], BG[2]), dtype=np.float32)
        bottom = np.array((BG_LIGHT[0], BG_LIGHT[1], BG_LIGHT[2]), dtype=np.float32)
        blend = np.linspace(0, 1, HEIGHT, dtype=np.float32)[:, None, None]
        rows = top[None, None, :] * (1 - blend) + bottom[None, None, :] * blend
        return np.repeat(rows.astype(np.uint8), WIDTH, axis=1)

    def _button(
        self,
        canvas: np.ndarray,
        box: tuple[int, int, int, int],
        label: str,
        action: str,
        primary: bool = False,
        enabled: bool = True,
        label_size: float = 0.65,
        label_thickness: int | None = None,
        outlined: bool = False,
    ) -> None:
        x1, y1, x2, y2 = box
        padding = self.BUTTON_HIT_PADDING
        hit_box = (
            x1 - padding,
            y1 - padding,
            x2 + padding,
            y2 + padding,
        )
        hit_x1, hit_y1, hit_x2, hit_y2 = hit_box
        mouse_hovered = (
            hit_x1 <= self._mouse_position[0] <= hit_x2
            and hit_y1 <= self._mouse_position[1] <= hit_y2
        )
        hand_hovered = (
            self._hand_position is not None
            and hit_x1 <= self._hand_position[0] <= hit_x2
            and hit_y1 <= self._hand_position[1] <= hit_y2
        )
        hovered = mouse_hovered or hand_hovered
        hover_amount = self._button_hover.get(action, 0.0)
        target_hover = 1.0 if hovered and enabled else 0.0
        if target_hover > hover_amount:
            hover_amount = max(
                target_hover * 0.65,
                hover_amount
                + (target_hover - hover_amount) * min(1.0, self._render_delta / 0.08),
            )
        else:
            hover_amount += (
                target_hover - hover_amount
            ) * min(1.0, self._render_delta / 0.12)
        self._button_hover[action] = hover_amount
        base_color = (
            BUTTON_PRIMARY
            if primary
            else CARD
            if outlined
            else CARD_LIGHT
        )
        hover_color = (
            BUTTON_PRIMARY_HOVER
            if primary
            else CARD_LIGHT
            if outlined
            else (94, 77, 63)
        )
        color = (
            self._mix_color(base_color, hover_color, hover_amount)
            if enabled
            else CARD
        )
        _rounded_rect(canvas, box, color, 16)
        if enabled and outlined:
            outline_color = self._mix_color(LINE, ACCENT, hover_amount)
            _rounded_rect(canvas, box, outline_color, 16, 1)
        text_thickness = (
            label_thickness
            if label_thickness is not None
            else 2
            if primary
            else 1
        )
        _center_text(
            canvas,
            label,
            ((x1 + x2) // 2, (y1 + y2) // 2),
            label_size,
            BG if primary and enabled else (TEXT if enabled else MUTED),
            text_thickness,
        )
        if enabled:
            self._buttons.append((hit_box, action))

    @staticmethod
    def _mix_color(
        start: tuple[int, int, int],
        end: tuple[int, int, int],
        amount: float,
    ) -> tuple[int, int, int]:
        return (
            round(start[0] * (1 - amount) + end[0] * amount),
            round(start[1] * (1 - amount) + end[1] * amount),
            round(start[2] * (1 - amount) + end[2] * amount),
        )

    @staticmethod
    def _game_action_button_boxes(
        panel: tuple[int, int, int, int],
    ) -> tuple[tuple[int, int, int, int], ...]:
        x1, y1, x2, y2 = panel
        horizontal_padding = 14
        horizontal_gap = 18
        button_widths = (100, 112, 100)
        available_width = x2 - x1 - horizontal_padding * 2
        row_width = sum(button_widths) + horizontal_gap * 2

        if available_width >= row_width:
            left = x1 + (x2 - x1 - row_width) // 2
            top = y2 - 17 - 48
            buttons = []
            for width in button_widths:
                buttons.append((left, top, left + width, top + 48))
                left += width + horizontal_gap
            return tuple(buttons)

        button_height = 27
        vertical_gap = 14
        bottom = y2 - 14
        top = bottom - button_height * 3 - vertical_gap * 2
        left = x1 + horizontal_padding
        right = x2 - horizontal_padding
        return tuple(
            (
                left,
                top + index * (button_height + vertical_gap),
                right,
                top + index * (button_height + vertical_gap) + button_height,
            )
            for index in range(3)
        )

    def _brand(self, canvas: np.ndarray) -> None:
        _rounded_rect(canvas, (48, 35, 96, 83), ACCENT, 15)
        _center_text(canvas, "R", (72, 59), 1.15, BG, 3)
        _text(canvas, "HANDPLAY", (110, 55), 0.8, TEXT, 2)
        _text(canvas, "COMPUTER VISION ARCADE", (111, 77), 0.42, MUTED, 1)
        cv2.line(canvas, (48, 105), (1312, 105), LINE, 1, cv2.LINE_AA)

    def _draw_hand_status(
        self, canvas: np.ndarray, hands_detected: int, navigation_gesture: str
    ) -> None:
        ready = hands_detected == 1
        status_color = SUCCESS if ready else DANGER
        _rounded_rect(canvas, (1000, 34, 1312, 92), CARD, 17)
        cv2.circle(canvas, (1026, 63), 8, status_color, -1, cv2.LINE_AA)
        label = (
            "Hand ready"
            if ready
            else "One hand only"
            if hands_detected > 1
            else "Show your hand"
        )
        _text(canvas, label, (1047, 61), 0.48, TEXT, 2)
        if navigation_gesture not in {"Unknown", ""}:
            _text(
                canvas,
                f"Gesture: {navigation_gesture}",
                (1047, 82),
                0.37,
                MUTED,
                1,
            )

    def _draw_hand_pointer(
        self, canvas: np.ndarray, position: tuple[int, int] | None
    ) -> None:
        if position is None:
            return
        cv2.circle(canvas, position, 17, ACCENT, 2, cv2.LINE_AA)
        cv2.circle(canvas, position, 4, ACCENT, -1, cv2.LINE_AA)

    def draw_interaction_debug(
        self,
        canvas: np.ndarray,
        hands_detected: int,
        index_tip: tuple[int, int] | None,
        cursor: tuple[int, int] | None,
        hovered_action: str | None,
        interaction_state: str,
        detection_ms: float = 0.0,
        fps: float = 0.0,
    ) -> None:
        """Show the shared interaction targets and current hand-button state."""
        for box, _action in self._buttons:
            cv2.rectangle(
                canvas,
                (box[0], box[1]),
                (box[2], box[3]),
                ACCENT,
                1,
                cv2.LINE_AA,
            )
        hovered_label = (
            "START GAME"
            if hovered_action == "start"
            else (hovered_action or "--").replace("_", " ").upper()
        )
        lines = (
            f"Hand: {'DETECTED' if hands_detected else 'LOST'}",
            f"Hand connection: {'ACTIVE' if cursor is not None else 'SEARCHING'}",
            f"Detection time: {detection_ms:.1f} ms",
            f"Index fingertip: {index_tip or '--'}",
            f"Cursor: {cursor or '--'}",
            f"FPS: {fps:.1f}",
            f"Hovered button: {hovered_label}",
            f"Click: {interaction_state}",
        )
        x, y = WIDTH - 390, 96
        _rounded_rect(
            canvas,
            (x - 8, y - 23, x + 365, y + (len(lines) - 1) * 21 + 10),
            BG,
            12,
        )
        for index, line in enumerate(lines):
            _text(canvas, line, (x, y + index * 21), 0.38, ACCENT, 1)

    def _draw_home(
        self, canvas: np.ndarray, video_frame: np.ndarray | None
    ) -> None:
        self._brand(canvas)
        _panel(canvas, (872, 176, 1286, 696), CARD, 28)
        _rounded_rect(canvas, (916, 221, 1242, 650), BG_LIGHT, 23)
        _center_text(canvas, "YOUR HAND IS THE CONTROLLER", (1079, 273), 0.53, MUTED, 1)
        self._gesture_card(canvas, (944, 318, 1213, 413), "ROCK", "Closed hand")
        self._gesture_card(canvas, (944, 432, 1213, 527), "PAPER", "Open hand")
        self._gesture_card(canvas, (944, 546, 1213, 641), "SCISSORS", "Two fingers")

        _panel(canvas, (96, 193, 814, 750), CARD, 28)
        _rounded_rect(canvas, (132, 232, 354, 269), (71, 59, 45), 18)
        _center_text(canvas, "Webcam gesture game", (243, 250), 0.45, ACCENT, 1)
        _rounded_rect(canvas, (610, 224, 790, 355), BG_LIGHT, 14)
        _text(canvas, "LIVE PREVIEW", (620, 215), 0.35, MUTED, 1)
        preview_box = (620, 230, 780, 345)
        if video_frame is not None:
            self._fit_frame(canvas, video_frame, preview_box)
        else:
            _center_text(canvas, "Starting...", (700, 288), 0.42, MUTED, 1)
        _text(canvas, "Rock • Paper", (135, 359), 1.35, TEXT, 3)
        _text(canvas, "& Scissors", (135, 426), 1.35, ACCENT, 3)
        _text(canvas, "Use your hand gestures to play", (139, 489), 0.62, MUTED, 1)
        _text(canvas, "against the computer.", (139, 521), 0.62, MUTED, 1)
        self._button(canvas, (136, 566, 438, 628), "Start Game", "start", True)
        self._button(canvas, (456, 566, 758, 628), "How to Play", "tutorial")
        self._button(canvas, (136, 645, 438, 707), "Settings", "settings")
        self._button(canvas, (456, 645, 758, 707), "Exit", "exit")
        _text(canvas, "Point at a button and hold briefly to select.", (138, 735), 0.45, MUTED, 1)

    def _gesture_card(
        self,
        canvas: np.ndarray,
        box: tuple[int, int, int, int],
        title: str,
        description: str,
    ) -> None:
        _rounded_rect(canvas, box, CARD, 15)
        _rounded_rect(canvas, (box[0] + 15, box[1] + 17, box[0] + 65, box[1] + 67), (75, 60, 44), 14)
        _draw_gesture_icon(canvas, (box[0] + 40, box[1] + 42), title.lower())
        _text(canvas, title.title(), (box[0] + 82, box[1] + 39), 0.65, TEXT, 2)
        _text(canvas, description, (box[0] + 82, box[1] + 67), 0.48, MUTED, 1)

    def _draw_tutorial(self, canvas: np.ndarray) -> None:
        self._brand(canvas)
        _text(canvas, "How to Play", (74, 164), 1.05, TEXT, 2)
        _text(canvas, "Five quick steps. Then you are ready to play.", (76, 202), 0.62, MUTED, 1)
        cards = (
            (76, "ROCK", "Close your hand"),
            (484, "PAPER", "Open your hand"),
            (892, "SCISSORS", "Raise two fingers"),
        )
        for x, title, description in cards:
            _rounded_rect(canvas, (x, 242, x + 374, 374), CARD, 22)
            _rounded_rect(canvas, (x + 20, 269, x + 78, 327), (77, 61, 44), 17)
            _draw_gesture_icon(canvas, (x + 49, 298), title.lower())
            _text(canvas, title, (x + 99, 294), 0.69, TEXT, 2)
            _text(canvas, description, (x + 99, 327), 0.49, MUTED, 1)

        instructions = (
            "1. Allow camera access when the game opens.",
            "2. Move one hand into the live camera area.",
            "3. Make a clear Rock, Paper, or Scissors pose.",
            "4. Wait for the countdown, then hold your pose.",
            "5. See both choices and the round result.",
        )
        _panel(canvas, (76, 408, 1284, 690), CARD, 23)
        _text(canvas, "Your first round", (111, 454), 0.6, ACCENT, 2)
        for index, instruction in enumerate(instructions):
            y = 500 + index * 34
            _center_text(canvas, str(index + 1), (123, y - 5), 0.48, ACCENT, 2)
            _text(canvas, instruction.split(". ", 1)[1], (153, y), 0.55, TEXT, 1)
        self._button(canvas, (76, 725, 278, 787), "Back", "home")
        self._button(canvas, (302, 725, 574, 787), "Start Game", "start", True)
        _text(canvas, "Tip: face your palm toward the camera and show only one hand.", (616, 763), 0.52, MUTED, 1)

    def _draw_settings(
        self, canvas: np.ndarray, sound_enabled: bool, volume_percent: int
    ) -> None:
        self._brand(canvas)
        _text(canvas, "Settings", (74, 164), 1.05, TEXT, 2)
        _text(canvas, "Personalize game feedback for this session.", (76, 202), 0.62, MUTED, 1)
        _panel(canvas, (330, 250, 1030, 650), CARD, 28)
        _text(canvas, "Sound effects", (400, 330), 0.7, TEXT, 2)
        _text(canvas, "Round results, countdown, and camera feedback.", (402, 369), 0.52, MUTED, 1)
        _rounded_rect(canvas, (402, 392, 958, 474), BG_LIGHT, 19)
        cv2.circle(canvas, (452, 433), 18, SUCCESS if sound_enabled else MUTED, -1, cv2.LINE_AA)
        _text(canvas, "ON" if sound_enabled else "OFF", (491, 443), 0.72, TEXT, 2)
        self._button(
            canvas,
            (766, 402, 922, 464),
            "Turn Off" if sound_enabled else "Turn On",
            "toggle_sound",
            primary=not sound_enabled,
        )
        _text(canvas, "Volume", (402, 523), 0.64, TEXT, 2)
        _text(canvas, f"{volume_percent}%", (866, 523), 0.56, ACCENT, 2)
        slider_box = (402, 540, 958, 599)
        self._sliders.append(slider_box)
        _rounded_rect(canvas, (422, 563, 938, 571), CARD_LIGHT, 4)
        fill_x = 422 + round(516 * volume_percent / 100)
        _rounded_rect(canvas, (422, 563, fill_x, 571), ACCENT, 4)
        cv2.circle(canvas, (fill_x, 567), 13, ACCENT, -1, cv2.LINE_AA)
        for index in range(11):
            x = 422 + index * 51
            cv2.circle(canvas, (x, 587), 2, MUTED, -1, cv2.LINE_AA)
        _text(canvas, "Click or point along the slider to set volume.", (402, 625), 0.42, MUTED, 1)
        self._button(canvas, (76, 725, 278, 787), "Back", "home")

    def _draw_message(
        self, canvas: np.ndarray, heading: str, detail: str, _variant: str
    ) -> None:
        self._brand(canvas)
        _panel(canvas, (310, 235, 1050, 650), CARD, 30)
        cv2.circle(canvas, (680, 345), 37, ACCENT, 4, cv2.LINE_AA)
        rotation = (time.monotonic() * 180) % 360
        cv2.ellipse(
            canvas,
            (680, 345),
            (19, 19),
            rotation,
            15,
            300,
            ACCENT,
            4,
            cv2.LINE_AA,
        )
        _center_text(canvas, heading, (680, 438), 0.94, TEXT, 2)
        _center_text(canvas, detail, (680, 489), 0.55, MUTED, 1)
        _text(canvas, "Camera access stays active for hand navigation.", (461, 584), 0.52, MUTED, 1)

    def _draw_error(self, canvas: np.ndarray, error_message: str) -> None:
        self._brand(canvas)
        _panel(canvas, (272, 214, 1088, 660), CARD, 28)
        _rounded_rect(canvas, (634, 263, 726, 355), (70, 55, 44), 24)
        _center_text(canvas, "!", (680, 309), 1.7, ACCENT, 3)
        _center_text(canvas, "CAMERA UNAVAILABLE", (680, 409), 0.86, TEXT, 2)
        _center_text(canvas, "Check that your webcam is connected and camera permission is enabled.", (680, 456), 0.49, MUTED, 1)
        _center_text(canvas, "Close other apps that may be using the camera, then try again.", (680, 485), 0.49, MUTED, 1)
        if error_message:
            _center_text(canvas, error_message[:100], (680, 526), 0.39, MUTED, 1)
        self._button(canvas, (429, 570, 667, 630), "Try Again", "retry", True)
        self._button(canvas, (693, 570, 909, 630), "Main Menu", "home")

    def _draw_game(
        self,
        canvas: np.ndarray,
        game: RockPaperScissorsGame,
        video_frame: np.ndarray | None,
        camera_active: bool,
        hands_detected: int,
        gesture: str,
        go_feedback: bool,
        result_animation_progress: float,
        gesture_pending: bool = False,
        gesture_hint: str = "Show your hand",
    ) -> None:
        self._brand(canvas)
        total_rounds = game.player_score + game.computer_score + game.draw_score
        round_number = total_rounds + (game.state != GameState.RESULT)
        _rounded_rect(canvas, (54, 124, 242, 161), CARD, 14)
        _text(canvas, f"Round {round_number}", (74, 149), 0.5, TEXT, 2)
        live_color = SUCCESS if camera_active else DANGER
        _rounded_rect(canvas, (1121, 124, 1306, 161), CARD, 14)
        cv2.circle(canvas, (1142, 142), 6, live_color, -1, cv2.LINE_AA)
        _text(canvas, "Camera live" if camera_active else "Camera off", (1157, 149), 0.45, live_color, 1)

        _panel(canvas, (54, 177, 895, 748), CARD, 23)
        _text(canvas, "Live camera", (83, 213), 0.51, TEXT, 2)
        hint_color = SUCCESS if gesture_hint == "Hand detected" else ACCENT
        _text(canvas, gesture_hint.upper(), (650, 213), 0.4, hint_color, 2)
        video_box = (78, 225, 871, 707)
        _rounded_rect(canvas, video_box, BG, 15)
        if video_frame is not None:
            self._fit_frame(canvas, video_frame, video_box)
        else:
            _center_text(canvas, "Waiting for video...", (475, 455), 0.6, MUTED, 1)
        _rounded_rect(
            canvas,
            video_box,
            SUCCESS if gesture_hint == "Hand detected" else ACCENT,
            15,
            2,
        )

        if game.state == GameState.COUNTDOWN:
            number = game.countdown_number(time.monotonic())
            countdown_label = "GET READY"
            self._countdown_overlay(canvas, number, countdown_label)
        elif game.state == GameState.PLAYING:
            self._countdown_overlay(
                canvas,
                0,
                "SHOOT!" if go_feedback else "SHOW YOUR HAND!",
            )

        if game.state == GameState.PLAYING and hands_detected == 0:
            _rounded_rect(canvas, (95, 627, 604, 661), (37, 48, 56), 11)
            _text(canvas, "No hand detected - move into the camera area", (111, 649), 0.38, TEXT, 1)

        side = (923, 177, 1306, 748)
        _panel(canvas, side, CARD, 23)
        _text(canvas, "Scoreboard", (954, 217), 0.48, MUTED, 2)
        _text(canvas, "You", (963, 255), 0.45, ACCENT, 2)
        _text(canvas, "Computer", (1158, 255), 0.45, COMPUTER, 2)
        _rounded_rect(canvas, (953, 270, 1104, 333), CARD_LIGHT, 15)
        _rounded_rect(canvas, (1123, 270, 1276, 333), CARD_LIGHT, 15)
        _center_text(canvas, str(game.player_score), (1028, 301), 1.05, TEXT, 2)
        _center_text(canvas, str(game.computer_score), (1199, 301), 1.05, COMPUTER, 2)
        _center_text(canvas, "DRAW " + str(game.draw_score), (1115, 356), 0.46, MUTED, 1)
        cv2.line(canvas, (953, 380), (1276, 380), LINE, 1, cv2.LINE_AA)

        _text(canvas, "Your gesture", (953, 414), 0.43, MUTED, 1)
        if game.state == GameState.RESULT:
            player_choice = game.player_choice
            computer_choice = game.computer_choice
        elif (
            game.state in {GameState.COUNTDOWN, GameState.PLAYING}
            and gesture in ("Rock", "Paper", "Scissors")
        ):
            player_choice = gesture
            computer_choice = "--"
        else:
            player_choice = "No hand" if hands_detected == 0 else "--"
            computer_choice = "--"
        self._choice_row(canvas, 953, 432, "YOU", player_choice, ACCENT)
        self._choice_row(canvas, 953, 500, "COMPUTER", computer_choice, COMPUTER)

        if game.state == GameState.RESULT:
            if game.last_result == "Player Wins":
                result_text, result_color = "You win!", SUCCESS
                result_mark = "+"
            elif game.last_result == "Computer Wins":
                result_text, result_color = "You lose", DANGER
                result_mark = "!"
            else:
                result_text, result_color = "Draw!", ACCENT
                result_mark = "="
            result_bg = (
                (48, 60, 49)
                if game.last_result == "Player Wins"
                else (48, 48, 76)
                if game.last_result == "Computer Wins"
                else (43, 62, 75)
            )
            eased = result_animation_progress * (2 - result_animation_progress)
            half_width = round(161.5 * (0.92 + 0.08 * eased))
            half_height = round(32.5 * (0.92 + 0.08 * eased))
            result_box = (
                1114 - half_width,
                607 - half_height,
                1114 + half_width,
                607 + half_height,
            )
            _rounded_rect(canvas, result_box, result_bg, 13)
            _center_text(
                canvas,
                f"{result_mark}  {result_text}",
                (1114, 607),
                0.68 + 0.12 * eased,
                result_color,
                2,
            )
        elif game.state == GameState.PLAYING:
            if hands_detected > 1:
                status = "Show only one hand."
            elif gesture_pending:
                status = "Hold still to lock in your choice."
            elif hands_detected == 1 and gesture == "Unknown":
                status = "Pose not recognized. Try again."
            elif player_choice != "--":
                status = f"Detected: {player_choice}"
            else:
                status = "Show your hand inside the camera area."
            _text(canvas, status[:37], (955, 611), 0.43, TEXT, 1)
        else:
            _text(canvas, game.status_message[:37], (955, 611), 0.41, MUTED, 1)

        if game.state == GameState.RESULT:
            button_text = "Round in progress"
            button_enabled = False
        elif game.last_result:
            button_text = "Next round"
            button_enabled = True
        else:
            button_text = "Start round"
            button_enabled = True
        main_menu_box, next_round_box, end_game_box = (
            self._game_action_button_boxes(side)
        )
        self._button(
            canvas,
            main_menu_box,
            "Main menu",
            "back",
            outlined=True,
            label_size=0.5,
            label_thickness=2,
        )
        self._button(
            canvas,
            next_round_box,
            button_text,
            "next_round",
            primary=True,
            enabled=button_enabled and game.state == GameState.WAITING,
            label_size=0.5,
            label_thickness=2,
        )
        self._button(
            canvas,
            end_game_box,
            "End game",
            "finish",
            outlined=True,
            label_size=0.5,
            label_thickness=2,
        )
        _text(canvas, "SPACE  Start round", (77, 786), 0.45, MUTED, 1)
        _text(canvas, "R  Reset score", (278, 786), 0.45, MUTED, 1)
        _text(canvas, "Q  End game", (443, 786), 0.45, MUTED, 1)

    def _fit_frame(
        self,
        canvas: np.ndarray,
        frame: np.ndarray,
        box: tuple[int, int, int, int],
    ) -> None:
        x1, y1, x2, y2 = box
        target_width, target_height = x2 - x1, y2 - y1
        source_height, source_width = frame.shape[:2]
        scale = min(target_width / source_width, target_height / source_height)
        image_width = max(1, round(source_width * scale))
        image_height = max(1, round(source_height * scale))
        left = x1 + (target_width - image_width) // 2
        top = y1 + (target_height - image_height) // 2
        resized = cv2.resize(
            frame,
            (image_width, image_height),
            interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
        )
        canvas[top:top + image_height, left:left + image_width] = resized

    def _countdown_overlay(self, canvas: np.ndarray, number: int, label: str) -> None:
        _rounded_rect(canvas, (330, 332, 620, 558), (29, 32, 34), 24)
        if number:
            scale = 3.6 + 0.16 * (time.monotonic() % 0.8)
            _center_text(canvas, str(number), (475, 420), scale, ACCENT, 4)
            _center_text(canvas, label, (475, 499), 0.58, TEXT, 2)
        else:
            _center_text(canvas, label, (475, 445), 0.77, ACCENT, 2)
            instruction = (
                "Show your hand now"
                if label == "SHOOT!"
                else "Hold one clear gesture"
            )
            _center_text(canvas, instruction, (475, 496), 0.49, TEXT, 1)

    def _choice_row(
        self,
        canvas: np.ndarray,
        x: int,
        y: int,
        label: str,
        choice: str,
        accent: tuple[int, int, int],
    ) -> None:
        _rounded_rect(canvas, (x, y, 1276, y + 54), CARD_LIGHT, 13)
        cv2.circle(canvas, (x + 22, y + 27), 5, accent, -1, cv2.LINE_AA)
        _text(canvas, label.title(), (x + 39, y + 23), 0.38, MUTED, 1)
        choice_size = 0.65 if label == "YOU" and choice in {
            "Rock",
            "Paper",
            "Scissors",
        } else 0.57
        _text(
            canvas,
            choice.title()
            if choice in {"Rock", "Paper", "Scissors"}
            else choice,
            (x + 39, y + 43),
            choice_size,
            TEXT,
            2,
        )
        if choice.lower() in ("rock", "paper", "scissors"):
            _draw_gesture_icon(
                canvas,
                (x + 286, y + 27),
                choice.lower(),
                accent,
            )

    def _draw_summary(self, canvas: np.ndarray, game: RockPaperScissorsGame) -> None:
        self._brand(canvas)
        _panel(canvas, (296, 150, 1064, 789), CARD, 28)
        _rounded_rect(canvas, (636, 190, 724, 278), (76, 61, 43), 24)
        _center_text(canvas, "RPS", (680, 234), 0.8, ACCENT, 2)
        if game.player_score > game.computer_score:
            headline, headline_color = "YOU WON THE GAME!", SUCCESS
        elif game.player_score < game.computer_score:
            headline, headline_color = "YOU LOSE", DANGER
        else:
            headline, headline_color = "FINAL DRAW", WARNING
        _center_text(canvas, headline, (680, 329), 0.95, headline_color, 2)
        _center_text(canvas, "Final score", (680, 371), 0.52, MUTED, 2)
        _rounded_rect(canvas, (428, 403, 932, 493), CARD_LIGHT, 20)
        _center_text(canvas, "YOU", (565, 428), 0.42, ACCENT, 1)
        _center_text(canvas, "COMPUTER", (795, 428), 0.42, COMPUTER, 1)
        _center_text(
            canvas,
            str(game.player_score),
            (565, 471),
            1.35,
            ACCENT,
            2,
        )
        _center_text(canvas, "-", (680, 471), 1.1, MUTED, 1)
        _center_text(
            canvas,
            str(game.computer_score),
            (795, 471),
            1.35,
            COMPUTER,
            2,
        )
        total = game.player_score + game.computer_score + game.draw_score
        if game.player_score > game.computer_score:
            winner = "Winner: YOU"
            winner_color = SUCCESS
        elif game.player_score < game.computer_score:
            winner = "Winner: COMPUTER"
            winner_color = DANGER
        else:
            winner = "Overall result: DRAW"
            winner_color = ACCENT
        _center_text(canvas, winner, (680, 536), 0.78, winner_color, 2)
        _text(canvas, f"Rounds played     {total}", (448, 591), 0.54, TEXT, 1)
        _text(canvas, f"Wins     {game.player_score}", (448, 627), 0.54, SUCCESS, 1)
        _text(canvas, f"Losses     {game.computer_score}", (688, 627), 0.54, DANGER, 1)
        _text(canvas, f"Draws     {game.draw_score}", (448, 663), 0.54, MUTED, 1)
        self._button(canvas, (420, 704, 660, 760), "Play again", "play_again", True)
        self._button(canvas, (684, 704, 924, 760), "Main menu", "home")
        self._button(canvas, (948, 704, 1052, 760), "Exit", "exit")
