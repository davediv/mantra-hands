"""
Vendors the hand model into the package, so the app never touches a CDN at
runtime.

This is not a tidiness thing. The app is used at an English camp in Pare on
shared wifi; a cold fetch of an 8 MB model is the difference between the camera
coming up in 300 ms and the room giving up on it.

    python3 -m mantra_hands.vendor

Run once, after cloning. It is the only thing in this project that ever opens a
socket, and it is not run automatically — an app that quietly downloads something
on first launch is exactly the failure this one was written to avoid. Everything
after it is offline forever.
"""

from __future__ import annotations

import shutil
import sys
import urllib.request

from .config import MODEL_CANDIDATES

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)


def main() -> int:
    destination = MODEL_CANDIDATES[0]

    for path in MODEL_CANDIDATES:
        if path.is_file():
            size = path.stat().st_size / 1e6
            if path == destination:
                print(f"vendor: hand_landmarker.task already present ({size:.1f} MB)")
                return 0
            # A sibling copy the web build already vendored. Copying beats
            # downloading it a second time over the same wifi.
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
            print(f"vendor: copied hand_landmarker.task from {path} ({size:.1f} MB)")
            return 0

    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"vendor: downloading hand_landmarker.task from {MODEL_URL}")
    temporary = destination.with_suffix(".part")
    try:
        with urllib.request.urlopen(MODEL_URL, timeout=60) as response:
            if response.status != 200:
                raise RuntimeError(f"model download failed: {response.status}")
            with temporary.open("wb") as out:
                shutil.copyfileobj(response, out)
        # Renamed only once it is whole, so an interrupted download can never
        # leave a truncated model that fails at load time with something obscure.
        temporary.replace(destination)
    except Exception as err:  # noqa: BLE001
        temporary.unlink(missing_ok=True)
        print(f"vendor: {err}", file=sys.stderr)
        return 1

    print(f"vendor: hand_landmarker.task -> {destination} "
          f"({destination.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
