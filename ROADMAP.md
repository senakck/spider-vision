# Roadmap

Spider Vision is built milestone by milestone. A milestone is only closed
after it has been tested on real hardware.

| # | Milestone | Goal | Status |
|---|---|---|---|
| M0 | Project Setup | Project skeleton, environment, tests, Git | ✅ Done |
| M1 | Camera | Webcam stream, FPS counter, ESC/Q to quit | ✅ Done |
| M2 | Hand Tracking | 21 hand landmarks with MediaPipe Tasks | ✅ Done |
| M3 | Gesture Recognition | OPEN_HAND, FIST, POINT, PINCH, SPIDER_MAN | ⏳ Planned |
| M4 | Mouse Control | Hand-controlled mouse, pinch to click | ⏳ Planned |
| M5 | Face Tracking | Face landmarks | ⏳ Planned |
| M6 | Spider-Man Mask | Mask overlay that follows the face | ⏳ Planned |
| M7 | Web Shooting | Shoot webs with the SPIDER_MAN gesture | ⏳ Planned |
| M8 | Visual Effects | Particles and web effects | ⏳ Planned |
| M9 | Pose Tracking | Body skeleton | ⏳ Planned |
| M10 | Game Mode | Gesture-controlled AR mini game | ⏳ Planned |
| M11 | Polish | Demo GIFs, docs, cleanup | ⏳ Planned |

## Technical decisions

Decisions are recorded here so the reasoning is not lost.

### M0 - Project Setup

- **Python 3.14.** mediapipe 1.0.1 officially lists 3.9-3.12, but ships a
  version-independent `py3-none` wheel. Installing it, importing it and loading
  the hand model were verified on 3.14 before committing to it.
- **`venv`** for the virtual environment: built into Python, no extra tools.
- **src layout** (`src/spider_vision/`): tests import the installed package,
  which catches packaging mistakes early.
- **Dependencies only in `pyproject.toml`**, as version ranges with an upper
  bound on the next major version (mediapipe is still alpha and its API changed
  between 0.10 and 1.0). Every directly imported package is listed explicitly.
- **`opencv-contrib-python`, never `opencv-python` as well.** mediapipe already
  depends on the contrib build; installing both breaks the `cv2` module.
- **Config as frozen dataclasses** in `config.py`: no extra dependency, type
  checked, and cannot be changed at runtime by accident. Each milestone adds
  its own section; thresholds are added only when they are measured.
- **Models downloaded by script**, not committed: keeps binaries out of Git
  history and always fetches the file from the official source.
- **MediaPipe Tasks API** (`mp.tasks.vision`), not the legacy `mp.solutions`.

### M1 - Camera

- **Logging:** standard library `logging`. Each module uses
  `logging.getLogger(__name__)`; logging is configured once in `__main__.py`
  from `AppConfig.log_level`. No `print` in library code.
- **Keyboard and window live in `__main__.py`**, not in the camera layer (the
  camera only delivers frames). ESC, Q and the window's X button all exit.
- **`Camera` is a context manager**, so the webcam is released even after an
  error. Its video source is injected (`source_factory`), which lets tests use
  a fake source instead of real hardware.
- **`Camera.read()` returns `None` on a missed frame** instead of raising; one
  dropped frame is normal. The main loop stops after
  `CameraConfig.max_read_failures` misses in a row (e.g. camera unplugged).
- **Frames are mirrored at capture** (`CameraConfig.mirror`), so the view
  behaves like a mirror. Note for M2: MediaPipe's left/right hand labels will
  be swapped on mirrored frames.
- **FPS is averaged over the last 30 frames** for a stable readout. The clock
  is injected, so tests run instantly without real time passing.
- **Baseline:** 30 FPS at 640x480 on the development laptop with no
  processing. M2 will be measured against this number.

### M2 - Hand Tracking

- **A MediaPipe-free domain model** (`vision/landmarks.py`): `Hand`,
  `Landmark` and the 21 named `HandLandmark`s. Gesture code reads
  `hand[HandLandmark.INDEX_FINGER_TIP]`, never raw MediaPipe indices.
- **Only `vision/hand_tracker.py` imports MediaPipe**, and
  `hands_from_result()` is the single place its results are converted. If the
  MediaPipe API changes again, only this file is affected.
- **Normalized coordinates for logic, pixels only for drawing.** Thresholds
  in normalized units work at any camera resolution.
- **VIDEO running mode**, not LIVE_STREAM: synchronous, so each frame's result
  belongs to that frame. Timestamps are forced to strictly increase
  (`MonotonicTimestamp`), as VIDEO mode requires.
- **Frames are converted BGR to RGB** before inference; OpenCV and MediaPipe
  use different channel orders.
- **Handedness is corrected for mirrored frames.** The Tasks API labels hands
  as they appear in an unmirrored image. Verified on a real webcam: without
  the swap, the right hand was labelled "Left".
- **`max_hands = 1`** for now; each extra hand adds tracking cost.
- **Performance:** still 29-30 FPS at 640x480 with tracking on, the same as
  the M1 baseline. The camera, not inference, is the bottleneck.
- **One integration test loads the real model**; it is skipped when the model
  has not been downloaded.
