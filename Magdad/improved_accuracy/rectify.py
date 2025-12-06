# rectify.py

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

from two_camera_coordinates.config import (
    STEREO_PARAMS_PATH,
    FRAME_WIDTH,
    FRAME_HEIGHT,
    BASELINE_REAL_MM,
)


def load_stereo_params(path: Path | str):
    data = np.load(path)
    K1 = data["K1"]
    d1 = data["d1"]
    K2 = data["K2"]
    d2 = data["d2"]
    R = data["R"]
    T = data["T"]
    baseline_est_mm = float(data["baseline_mm"])
    img_width = int(data["img_width"])
    img_height = int(data["img_height"])
    rms = float(data["rms"])

    if BASELINE_REAL_MM is not None and baseline_est_mm > 0:
        scale = BASELINE_REAL_MM / baseline_est_mm
        T = T * scale
        baseline_mm = BASELINE_REAL_MM
    else:
        baseline_mm = baseline_est_mm

    return K1, d1, K2, d2, R, T, baseline_mm, (img_width, img_height), rms


def init_rectification():
    (
        K1,
        d1,
        K2,
        d2,
        R,
        T,
        baseline_mm,
        (img_width, img_height),
        rms,
    ) = load_stereo_params(STEREO_PARAMS_PATH)

    img_size = (img_width, img_height)

    if img_width != FRAME_WIDTH or img_height != FRAME_HEIGHT:
        print(
            f"[WARN] Calibration image size {img_size} differs from "
            f"current frame size {(FRAME_WIDTH, FRAME_HEIGHT)}. "
            f"Make sure they match, or recalibrate at the current resolution."
        )

    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        K1,
        d1,
        K2,
        d2,
        img_size,
        R,
        T,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0,
    )

    map1x, map1y = cv2.initUndistortRectifyMap(
        K1, d1, R1, P1, img_size, cv2.CV_32FC1
    )
    map2x, map2y = cv2.initUndistortRectifyMap(
        K2, d2, R2, P2, img_size, cv2.CV_32FC1
    )

    return {
        "K1": K1,
        "d1": d1,
        "K2": K2,
        "d2": d2,
        "R": R,
        "T": T,
        "baseline_mm": baseline_mm,
        "img_size": img_size,
        "rms": rms,
        "R1": R1,
        "R2": R2,
        "P1": P1,
        "P2": P2,
        "Q": Q,
        "map1x": map1x,
        "map1y": map1y,
        "map2x": map2x,
        "map2y": map2y,
    }


def rectify_pair(
    frame_left: np.ndarray,
    frame_right: np.ndarray,
    maps: dict,
) -> Tuple[np.ndarray, np.ndarray]:
    map1x = maps["map1x"]
    map1y = maps["map1y"]
    map2x = maps["map2x"]
    map2y = maps["map2y"]

    rect_left = cv2.remap(
        frame_left, map1x, map1y, interpolation=cv2.INTER_LINEAR
    )
    rect_right = cv2.remap(
        frame_right, map2x, map2y, interpolation=cv2.INTER_LINEAR
    )
    return rect_left, rect_right
