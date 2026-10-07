# Rock Paper Scissors — Hand Gesture Game

Play Rock Paper Scissors against the computer using a webcam. MediaPipe finds
hand landmarks, the gesture module classifies a hand pose, and OpenCV displays
the camera image, landmarks, countdown, choices, result, and score. The game
initializes the webcam at launch so hand gestures can navigate the welcome screen.

## Features

- Real-time, mirrored webcam preview with hand landmarks; the game play area
  shows the full camera frame without stretching or center-cropping it.
- Camera capture and MediaPipe inference run on a dedicated worker that
  publishes the latest frame and landmarks without blocking interface updates.
- Welcome screen, visual tutorial, camera-loading/error screens, and final summary.
- Large in-game countdown, live gesture feedback, clear round results, and score.
- Segoe UI typography, high-contrast slate-blue cards, cyan player and purple
  computer accents, and large menu controls.
- Explicit player/computer labels on the scoreboard and final-score summary.
- Camera-backed menu pointer with immediate button highlighting and selection
  after hovering over a button for one second; no click gesture is required.
  Moving away cancels the hold, and leaving the selected button rearms it. A
  fist goes back. Mouse and keyboard remain available.
- Enlarged button hit areas remain separated from neighboring controls, and
  the game camera panel provides a taller, more comfortable hand-play view.
- Lightweight landmark smoothing reduces tracking jitter while preserving
  quick movement; a brief missed-frame grace keeps re-acquired tracking smooth.
  A three-frame gesture consensus ignores single-frame noise and follows a
  changed pose after two matching frames. The fixed-size interface scales
  proportionally into resized windows while keeping mouse coordinates aligned.
- Original WAV effects for hover, clicks, countdown, gestures, round results,
  and game completion, with a session-only master volume and mute control.
- Clickable buttons plus keyboard controls throughout the game.
- Recognizes Rock, Paper, and Scissors; other poses are not guessed.
- Rejects frames with no hand or more than one hand.
- Three-second 3-2-1-SHOOT countdown; the camera keeps sampling throughout so
  a stable hand pose at SHOOT is captured immediately.
- First-to-three-wins matches, with the first round and every later round
  starting automatically; after the winner is shown, use the same one-second
  hand-hover selection to play again, return home, or exit.
- Score tracking and a reset control.
- Separate modules for webcam detection, gesture recognition, navigation,
  sound, game rules, and UI.
- Unit tests for gesture classification, game rules, round scoring, and UI rendering.

## Technologies

- Python 3.11
- OpenCV (`opencv-contrib-python` provides the `cv2` module)
- MediaPipe Hands
- NumPy
- Pillow (Segoe UI text rendering)
- Python standard library (`random`, `time`, `enum`, `unittest`, `wave`, and Windows `winsound`)

Dependencies are pinned in `requirements.txt` to versions with Python 3.11
support. The MediaPipe version uses its `solutions.hands` API.

## Project structure

```text
Rock-Paper-Scissors/
├── main.py
├── game.py
├── hand_detector.py
├── gesture.py
├── ui.py
├── sound.py
├── navigation.py
├── assets/
│   └── sounds/
│       ├── click.wav
│       ├── hover.wav
│       ├── camera_ready.wav
│       ├── hand_confirmed.wav
│       ├── countdown.wav
│       ├── countdown_final.wav
│       ├── go.wav
│       ├── win.wav
│       ├── lose.wav
│       ├── draw.wav
│       ├── final_win.wav
│       ├── final_lose.wav
│       └── final_draw.wav
├── requirements.txt
├── README.md
└── tests/
    ├── test_game.py
    ├── test_ui.py
    ├── test_sound.py
    └── test_navigation.py
```

OpenCV handles camera and interface drawing; Pillow renders readable Segoe UI
type. Pillow uses its bundled DejaVu Sans fallback when Segoe UI is unavailable.
Sound effects are bundled as small, original mono WAV files under
`assets/sounds/` and play asynchronously through Windows' built-in `winsound`
module. The master volume scales the PCM assets without adding an audio package.
The detector provides hand landmarks rather than a calibrated gesture-confidence
score, so the interface reports recognized gestures without inventing a
confidence percentage.

## Installation on Windows

Open PowerShell or Command Prompt in the project folder:

```powershell
cd C:\Users\USER\Desktop\Algorithm-Final\Rock-Paper-Scissors
py -3.11 -m venv .venv
```

Activate the virtual environment.

PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Command Prompt:

```bat
.venv\Scripts\activate.bat
```

Install the requirements:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If `py -3.11` is not recognized, install Python 3.11 from python.org and enable
the Python Launcher during setup. Do not substitute a different Python version
without checking that MediaPipe provides a compatible wheel for it.

## How to run

From the activated environment and project folder:

```powershell
python main.py
```

The app requests camera access at launch so hand navigation works on the Home
screen. The live hand status is shown in the header. Select **Settings** to
toggle sound effects and adjust master volume for the current session.

Move the hand cursor over a button and keep it there for one second to select
it, just like clicking; no special click pose is needed. Moving away cancels
the hold, and leaving the selected button rearms it. Hold a fist steadily to
go back (or end a game to its summary). Hand navigation is active in menus,
the round-waiting/result states, and the final summary; it is disabled while a
Rock/Paper/Scissors choice is being detected. Mouse and keyboard remain
available as backups.

## How to play

1. Allow camera access when the game opens. On Home, move the hand cursor over
   **START GAME** and keep it there for one second, or use the mouse/ENTER.
2. The first round starts automatically. Show your pose during the **3, 2, 1**
   countdown and hold it through **SHOOT!** for immediate capture.
3. Keep one clear Rock, Paper, or Scissors pose visible to the camera.
4. The game records the first recognized pose, chooses for the computer, and
   shows the winner and updated score.
5. After each result pause, the next round starts automatically. The first
   player to win three rounds wins the match.
6. On the winner screen, hover over **PLAY AGAIN**, **MAIN MENU**, or **EXIT**
   for one second to choose what to do next.

Hold your palm toward the camera and keep it upright for the most reliable
finger checks. Only one hand should be visible.

## Keyboard controls

- **ENTER** — Start a game from the home screen; retry camera; or play again.
- **H** — Open the tutorial from the home screen or return to the main menu
  from the final summary.
- **Settings** — Turn sound effects on/off and set master volume for this session.
- **Hand** — Move the hand cursor over a button to highlight it immediately;
  keep it there for one second to select. Moving away cancels the hold, and
  leaving the selected button rearms it. A steady fist returns to the previous menu.
- **SPACE** — Start a game from the home screen or play again from the final
  summary. Rounds start automatically during a match.
- **R** — Reset the match score; the next round starts automatically.
- **Q** — End the current game and show its summary; quit from the home screen
  or final summary.
- **ESC** — Return to the previous screen, end the current game, or return to
  the main menu from the summary.
- **D** — Show or hide hand connection, cursor, detection-time, and FPS diagnostics.
- **Mouse** — Click the on-screen buttons.

## Testing

Run the built-in tests from the project folder:

```powershell
python -m unittest discover -s tests -v
```

Manual checklist:

- [ ] Python 3.11 works
- [ ] Virtual environment activates
- [ ] OpenCV imports and opens a window
- [ ] MediaPipe imports and processes a frame
- [ ] NumPy imports
- [ ] Webcam opens and displays video
- [ ] Home preview shows the camera and detected hand landmarks
- [ ] Welcome and tutorial screens respond to mouse and keyboard
- [ ] Settings can turn sound effects on and off
- [ ] Settings volume slider changes the WAV playback level
- [ ] Home opens after camera initialization and shows hand status
- [ ] The cursor appears at the index fingertip on first detection; brief hand loss
  preserves the cursor and longer loss reconnects at the new fingertip position
- [ ] Hovering the hand cursor highlights a button; a one-second dwell selects once without a click pose
- [ ] A stable fist goes back once and respects the cooldown
- [ ] Home EXIT closes the app
- [ ] Mouse and keyboard remain usable as backups
- [ ] Start/tutorial/next-round/play-again/menu/retry buttons play one soft click
- [ ] Camera-ready effect plays after successful initialization
- [ ] Countdown beeps progress through 3, 2, 1, and GO exactly once
- [ ] Hand-confirmation effect does not repeat while a hand remains visible
- [ ] Win, lose, and draw effects match the round result
- [ ] Final win, loss, and draw effects match the overall score
- [ ] Camera initializes at launch and stays active across screen changes
- [ ] Camera error offers retry and main-menu actions
- [ ] Hand landmarks appear
- [ ] Rock is recognized
- [ ] Paper is recognized
- [ ] Scissors is recognized
- [ ] Computer choice is Rock, Paper, or Scissors
- [ ] Winner calculation is correct
- [ ] Scores update after a round
- [ ] Countdown and live gesture/no-hand feedback are clear
- [ ] Result names both choices and the winner
- [ ] First round and later rounds start automatically without selecting a button
- [ ] Match ends when either side gets three wins; draws do not count as wins
- [ ] Winner screen offers Play Again, Main Menu, and Exit through one-second hand dwell
- [ ] Ending a game displays final score, wins, losses, and draws
- [ ] Play Again resets the score and returns to gameplay without restarting the camera
- [ ] R resets the score
- [ ] Main Menu returns to the welcome screen
- [ ] Q ends the game; the window close button exits cleanly

## Common errors and solutions

- **The webcam could not be opened at startup:** Check the camera connection
  and Windows camera privacy settings. Close other applications that may be
  using it; retry or use the mouse/keyboard on the error screen.
- **The webcam opens but no image appears:** Try another camera if available.
  The program uses camera index `0`; a different index may be needed on systems
  with multiple cameras.
- **PowerShell blocks activation:** Run
  `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` in PowerShell, or use
  Command Prompt and `activate.bat`.
- **No hand detected:** Improve the lighting, keep your whole hand in view, and
  move closer to the camera.
- **Multiple hands detected:** Keep only one hand in the camera frame.
- **Gesture not recognized:** Hold a distinct, upright pose with the palm facing
  the camera. The program waits for a valid gesture instead of choosing one for
  you.
- **A package cannot be installed:** Confirm Python is 64-bit Python 3.11,
  activate this project's `.venv`, and run the installation commands again.
- **MediaPipe import or API error:** Confirm that the project's pinned
  `mediapipe==0.10.21` and `opencv-contrib-python==4.11.0.86` were installed in
  the active virtual environment.
- **UI font not rendered as expected:** Reinstall the requirements to ensure
  Pillow is available; the renderer falls back to DejaVu Sans if Segoe UI is
  missing from Windows.
