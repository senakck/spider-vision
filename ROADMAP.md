# Roadmap

Spider Vision is built milestone by milestone. A milestone is only closed
after it has been tested on real hardware.

| # | Milestone | Goal | Status |
|---|---|---|---|
| M0 | Project Setup | Project skeleton, environment, tests, Git | ✅ Done |
| M1 | Camera | Webcam stream, FPS counter, ESC/Q to quit | ⏳ Planned |
| M2 | Hand Tracking | 21 hand landmarks with MediaPipe Tasks | ⏳ Planned |
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

### Decided for M1

- **Logging:** standard library `logging`. Each module uses
  `logging.getLogger(__name__)`; logging is configured once at the entry point
  from `AppConfig.log_level`. No `print` in library code.
- **ESC/Q exit:** handled in the main loop at the entry point, not in the
  camera layer (the camera only delivers frames). Camera release is guaranteed
  with `try/finally`, so the webcam is freed even after an error.
