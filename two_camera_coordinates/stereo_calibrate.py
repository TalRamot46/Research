# stereo_calibrate.py

import glob
from pathlib import Path
import numpy as np
import cv2

from config import (
    CHESSBOARD_ROWS,
    CHESSBOARD_COLS,
    SQUARE_SIZE_MM,
    CALIB_LEFT_DIR,
    CALIB_RIGHT_DIR,
    STEREO_PARAMS_PATH,
)

def prepare_object_points() -> np.ndarray:
    objp = np.zeros((CHESSBOARD_ROWS * CHESSBOARD_COLS, 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHESSBOARD_COLS, 0:CHESSBOARD_ROWS].T.reshape(-1, 2)
    objp *= SQUARE_SIZE_MM
    return objp

def find_corners(gray, pattern_size, criteria):
    flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH
        + cv2.CALIB_CB_NORMALIZE_IMAGE
        + cv2.CALIB_CB_FAST_CHECK
    )

    ret, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
    if not ret:
        return False, None

    corners_refined = cv2.cornerSubPix(
        gray,
        corners,
        winSize=(11, 11),
        zeroZone=(-1, -1),
        criteria=criteria,
    )
    return True, corners_refined


def load_image_pairs():
    left_paths = sorted(glob.glob(str(CALIB_LEFT_DIR / "left_*.png")))
    right_paths = sorted(glob.glob(str(CALIB_RIGHT_DIR / "right_*.png")))

    if not left_paths or not right_paths:
        raise RuntimeError("No calibration images found. Did you run capture_stereo_pairs.py?")

    if len(left_paths) != len(right_paths):
        raise RuntimeError("Left/right image counts differ. Ensure pairs are consistent.")

    return [Path(p) for p in left_paths], [Path(p) for p in right_paths]

def stereo_calibrate():
    objp = prepare_object_points()
    objpoints = []
    imgpoints_left = []
    imgpoints_right = []

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        1e-6,
    )

    pattern_size = (CHESSBOARD_COLS, CHESSBOARD_ROWS)
    left_paths, right_paths = load_image_pairs()

    example = cv2.imread(str(left_paths[0]))
    if example is None:
        raise RuntimeError(f"Cannot read example image {left_paths[0]}")
    gray_example = cv2.cvtColor(example, cv2.COLOR_BGR2GRAY)
    img_size = gray_example.shape[::-1]

    for lp, rp in zip(left_paths, right_paths):
        img_l = cv2.imread(str(lp))
        img_r = cv2.imread(str(rp))
        if img_l is None or img_r is None:
            print(f"[WARNING] Failed to read {lp} or {rp}")
            continue

        gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

        found_l, corners_l = find_corners(gray_l, pattern_size, criteria)
        found_r, corners_r = find_corners(gray_r, pattern_size, criteria)

        if found_l and found_r:
            objpoints.append(objp)
            imgpoints_left.append(corners_l)
            imgpoints_right.append(corners_r)
        else:
            print(f"[INFO] Skipping pair (no corners): {lp.name}, {rp.name}")

    if len(objpoints) < 5:
        raise RuntimeError("Not enough valid pairs with detected corners (need ~5 or more).")

    ret_l, K1, d1, _, _ = cv2.calibrateCamera(
        objpoints, imgpoints_left, img_size, None, None
    )
    ret_r, K2, d2, _, _ = cv2.calibrateCamera(
        objpoints, imgpoints_right, img_size, None, None
    )

    flags = cv2.CALIB_FIX_INTRINSIC
    stereo_criteria = (
        cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS,
        100,
        1e-5,
    )

    rms, K1, d1, K2, d2, R, T, E, F = cv2.stereoCalibrate(
        objpoints,
        imgpoints_left,
        imgpoints_right,
        K1,
        d1,
        K2,
        d2,
        img_size,
        criteria=stereo_criteria,
        flags=flags,
    )

    baseline_mm = float(np.linalg.norm(T))

    np.savez(
        STEREO_PARAMS_PATH,
        K1=K1,
        d1=d1,
        K2=K2,
        d2=d2,
        R=R,
        T=T,
        E=E,
        F=F,
        img_width=img_size[0],
        img_height=img_size[1],
        rms=rms,
        baseline_mm=baseline_mm,
    )

    print("[INFO] Stereo calibration complete.")
    print(f"       RMS reprojection error: {rms:.4f}")
    print("       K1:")
    print(K1)
    print("       K2:")
    print(K2)
    print(f"       Baseline: {baseline_mm:.2f} mm")

if __name__ == "__main__":
    stereo_calibrate()
