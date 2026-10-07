"""Check Python/CV dependencies and optionally preview webcam input."""

from __future__ import annotations

import argparse
import importlib
import sys
from dataclasses import dataclass
from importlib import metadata
from typing import Any


REQUIRED_MODULES = (
    ("OpenCV", "cv2", "opencv-python"),
    ("MediaPipe", "mediapipe", "mediapipe"),
    ("NumPy", "numpy", "numpy"),
)
OPTIONAL_MODULES = (
    ("cvzone (optional hand-tracking helper)", "cvzone", "cvzone"),
)


@dataclass
class ModuleCheck:
    """Container for dependency status details."""

    label: str
    module_name: str
    distribution_name: str
    module: Any | None = None
    version: str | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.module is not None

    def describe(self) -> str:
        if self.ok:
            return f"{self.label}: PASS - version {self.version}"
        error = self.error or "unknown error"
        return f"{self.label}: FAIL - {error}"


def check_module(label: str, module_name: str, distribution_name: str) -> ModuleCheck:
    """Import a module and return the installed version when available."""
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # pragma: no cover - exercised during environment checks.
        return ModuleCheck(
            label=label,
            module_name=module_name,
            distribution_name=distribution_name,
            error=f"{type(exc).__name__}: {exc}",
        )

    try:
        version = metadata.version(distribution_name)
    except metadata.PackageNotFoundError:
        version = getattr(module, "__version__", "unknown")

    return ModuleCheck(
        label=label,
        module_name=module_name,
        distribution_name=distribution_name,
        module=module,
        version=str(version),
    )


def preview_webcam(cv2_module: Any | None) -> bool:
    """Open the default camera and show a short live preview until the user exits."""
    title = "Webcam test — press Q or Esc to close"

    if cv2_module is None:
        print("Webcam: FAIL - OpenCV is unavailable, so the camera cannot be tested.")
        return False

    camera = cv2_module.VideoCapture(0)
    try:
        if not camera.isOpened():
            print(
                "Webcam: FAIL - camera 0 could not be opened. "
                "Check camera connection, permissions, and whether another app is using it."
            )
            return False

        success, frame = camera.read()
        if not success or frame is None:
            print("Webcam: FAIL - camera opened, but no video frame could be read.")
            return False

        try:
            cv2_module.imshow(title, frame)
            cv2_module.waitKey(1)
        except cv2_module.error as exc:
            print(f"Webcam: FAIL - OpenCV could not display a window: {exc}")
            return False

        print("Webcam: PASS - preview window opened. Press Q or Esc to close.")
        while True:
            success, frame = camera.read()
            if not success or frame is None:
                print("Webcam: FAIL - video stopped before the preview ended.")
                return False

            cv2_module.imshow(title, frame)
            key = cv2_module.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                return True

            try:
                if cv2_module.getWindowProperty(title, cv2_module.WND_PROP_VISIBLE) < 1:
                    return True
            except cv2_module.error:
                pass
    except cv2_module.error as exc:
        print(f"Webcam: FAIL - OpenCV camera error: {exc}")
        return False
    finally:
        camera.release()
        cv2_module.destroyAllWindows()


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line interface for the environment checker."""
    parser = argparse.ArgumentParser(
        description="Check if the computer has the required Python, OpenCV, and MediaPipe dependencies.",
    )
    parser.add_argument(
        "--skip-webcam",
        action="store_true",
        help="Skip the live webcam preview and only validate the installed packages.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose per-module output while still printing the summary.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    print(f"Python: PASS - {sys.version.split()[0]} ({sys.executable})")
    if sys.version_info[:2] > (3, 12):
        print(
            "Note: MediaPipe currently lists Python support through 3.12; "
            "use Python 3.12 for the most compatible setup."
        )

    required = {module_name: check_module(label, module_name, distribution_name) for label, module_name, distribution_name in REQUIRED_MODULES}
    optional = [check_module(label, module_name, distribution_name) for label, module_name, distribution_name in OPTIONAL_MODULES]

    if not args.quiet:
        for item in required.values():
            print(item.describe())
        for item in optional:
            print(item.describe())

    print("Other required libraries: PASS - no additional required libraries are declared in this project; cvzone is optional.")

    webcam_ok = True if args.skip_webcam else preview_webcam(required["cv2"].module)
    checks_ok = all(item.ok for item in required.values()) and webcam_ok

    print("\nSummary")
    print(f"Python: PASS")
    print(f"OpenCV: {'PASS' if required['cv2'].ok else 'FAIL'}")
    print(f"MediaPipe: {'PASS' if required['mediapipe'].ok else 'FAIL'}")
    print(f"NumPy: {'PASS' if required['numpy'].ok else 'FAIL'}")
    print(f"Webcam: {'PASS' if webcam_ok else 'FAIL'}")
    print("Other required libraries: PASS")
    return 0 if checks_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
