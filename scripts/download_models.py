"""Download the MediaPipe model files used by Spider Vision.

Models are not committed to the repository. Run this once after cloning:

    python scripts/download_models.py

Files are saved to ``models/`` (git-ignored). Existing files are skipped.
Models are published by Google under the Apache 2.0 license.
"""

from pathlib import Path
from urllib.request import urlopen

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

# file name -> official download URL. Later milestones add face/pose models here.
MODELS: dict[str, str] = {
    "hand_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/"
        "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
    ),
}


def download(name: str, url: str) -> None:
    target = MODELS_DIR / name
    if target.exists():
        print(f"[skip] {name} already exists")
        return

    # Download to a temporary file first, so an interrupted download
    # never leaves a broken model file behind.
    partial = target.with_suffix(target.suffix + ".part")
    print(f"[download] {name} ...")
    with urlopen(url, timeout=60) as response:
        partial.write_bytes(response.read())
    partial.replace(target)

    size_mb = target.stat().st_size / 1_000_000
    print(f"[ok] {name} ({size_mb:.1f} MB)")


def main() -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    for name, url in MODELS.items():
        download(name, url)


if __name__ == "__main__":
    main()
