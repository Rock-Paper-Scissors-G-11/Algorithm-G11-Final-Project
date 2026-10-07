"""Check Python/CV dependencies and preview webcam input."""

import importlib
import sys
from importlib import metadata


REQUIRED_MODULES = (
    ("OpenCV", "cv2", "opencv-python"),
    ("MediaPipe", "mediapipe", "mediapipe"),
    ("NumPy", "numpy", "numpy"),
)
OPTIONAL_MODULES = (
    ("cvzone (optional hand-tracking helper)", "cvzone", "cvzone"),
)


def check_module(label, module_name, distribution_name):
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        print(f"{label}: FAIL — {type(exc).__name__}: {exc}")
        return None

    try:
        version = metadata.version(distribution_name)
    except metadata.PackageNotFoundError:
        version = getattr(module, "__version__", "unknown")
    print(f"{label}: PASS — version {version}")
    return module


def preview_webcam(cv2):
    if cv2 is None:
        print("Webcam: FAIL — OpenCV is unavailable, so the camera cannot be tested.")
        return False

    camera = cv2.VideoCapture(0)
    try:
        if not camera.isOpened():
            print(
                "Webcam: FAIL — camera 0 could not be opened. "
                "Check camera connection, permissions, and whether another app is using it."
            )
            return False

        success, frame = camera.read()
        if not success or frame is None:
            print("Webcam: FAIL — camera opened, but no video frame could be read.")
            return False

        try:
            cv2.imshow("Webcam test — press Q or Esc to close", frame)
            cv2.waitKey(1)
        except cv2.error as exc:
            print(f"Webcam: FAIL — OpenCV could not display a window: {exc}")
            return False

        print("Webcam: PASS — preview window opened. Press Q or Esc to close.")
        while True:
            success, frame = camera.read()
            if not success or frame is None:
                print("Webcam: FAIL — video stopped before the preview ended.")
                return False

            cv2.imshow("Webcam test — press Q or Esc to close", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                return True

            try:
                if cv2.getWindowProperty(
                    "Webcam test — press Q or Esc to close",
                    cv2.WND_PROP_VISIBLE,
                ) < 1:
                    return True
            except cv2.error:
                pass
    except cv2.error as exc:
        print(f"Webcam: FAIL — OpenCV camera error: {exc}")
        return False
    finally:
        camera.release()
        cv2.destroyAllWindows()


def main():
    print(f"Python: PASS — {sys.version.split()[0]} ({sys.executable})")
    if sys.version_info[:2] > (3, 12):
        print(
            "Note: MediaPipe currently lists Python support through 3.12; "
            "use Python 3.12 for the most compatible setup."
        )

    modules = {}
    for label, module_name, distribution_name in REQUIRED_MODULES:
        modules[module_name] = check_module(label, module_name, distribution_name)

    for label, module_name, distribution_name in OPTIONAL_MODULES:
        check_module(label, module_name, distribution_name)

    print(
        "Other required libraries: PASS — no additional required libraries "
        "are declared in this project; cvzone above is optional."
    )

    webcam_ok = preview_webcam(modules["cv2"])
    checks_ok = all(modules.values()) and webcam_ok
    print("\nSummary")
    print(f"Python: PASS")
    print(f"OpenCV: {'PASS' if modules['cv2'] else 'FAIL'}")
    print(f"MediaPipe: {'PASS' if modules['mediapipe'] else 'FAIL'}")
    print(f"NumPy: {'PASS' if modules['numpy'] else 'FAIL'}")
    print(f"Webcam: {'PASS' if webcam_ok else 'FAIL'}")
    print("Other required libraries: PASS")
    return 0 if checks_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
