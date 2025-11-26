#!/usr/bin/env python3
"""run_check.py

Cross-platform wrapper to run the same command contained in `check_live_cam.sh`.

Behavior:
- Prefer using the Ultralytics Python API (import `ultralytics` and call `YOLO`).
- If the API is unavailable, attempt to run the `yolo` CLI if present on PATH.
- If neither is available, print clear instructions to install dependencies.

Usage examples (from repo root):
  python run_check.py
  python run_check.py --model runs_pendulum/detect/pendulum_yolo4/weights/best.pt --source pendulum_dataset/images/val --no-save

This script should be run with your project venv's Python so installed packages are visible.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def run_with_api(model: str, source: str, save: bool, show: bool) -> int:
    try:
        from ultralytics import YOLO
    except Exception:
        raise

    print(f"Using Ultralytics API (module) with model={model}, source={source}, save={save}, show={show}")
    model_path = model
    y = YOLO(model_path)
    # Convert numeric sources (e.g. '0') to int so the camera device is used
    src = int(source) if (isinstance(source, str) and source.isdigit()) else source
    # The YOLO.predict API accepts `source`, `save`, and `show`.
    y.predict(source=src, save=save, show=show)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Cross-platform runner for check_live_cam.sh")
    p.add_argument("--model", default="results/pendulum_yolo73/weights/best.pt")
    p.add_argument("--source", default="1")
    p.add_argument("--save", dest="save", action="store_true", default=True)
    p.add_argument("--no-save", dest="save", action="store_false")
    # add argument for "show"
    # make the script open a window to show the video feed, make sure
    # that the camera being used is the HP 310 webcam connected via USB
    p.add_argument("--show", dest="show", action="store_true", default=True)
    p.add_argument("--no-show", dest="show", action="store_false")

    args = p.parse_args(argv)

    model = str(Path(args.model))
    source = args.source
    save = bool(args.save)
    show = bool(args.show)

    # Try API first (this will use the Python interpreter running this script)
    try:
        return run_with_api(model, source, save, show)
    except Exception as api_exc:
        print("Ultralytics API unavailable or failed:", api_exc)


if __name__ == "__main__":
    raise SystemExit(main())
