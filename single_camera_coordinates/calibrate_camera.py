# calibrate_webcam.py

import glob
from pathlib import Path
import numpy as np
import cv2

from config import (
    CHESSBOARD_ROWS,
    CHESSBOARD_COLS,
    SQUARE_SIZE_MM,
    CALIB_IMAGES_DIR,
    CAMERA_PARAMS_PATH,
)

def prepare_object_points() -> np.ndarray:
    objp = np.zeros((CHESSBOARD_ROWS * CHESSBOARD_COLS, 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHESSBOARD_COLS, 0:CHESSBOARD_ROWS].T.reshape(-1, 2)
    objp *= SQUARE_SIZE_MM
    return objp

def find_corners_in_image(image_path: Path, criteria) -> tuple[bool, np.ndarray | None]:
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"[WARNING] Could not read image {image_path}")
        return False, None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    pattern_size = (CHESSBOARD_COLS, CHESSBOARD_ROWS)
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

    if not ret:
        print(f"[INFO] Chessboard not found in {image_path}")
        return False, None

    corners_refined = cv2.cornerSubPix(
        gray,
        corners,
        winSize=(11, 11),
        zeroZone=(-1, -1),
        criteria=criteria,
    )
    return True, corners_refined

def calibrate_camera() -> None:
    image_paths = sorted(glob.glob(str(CALIB_IMAGES_DIR / "*.png")))
    if not image_paths:
        raise RuntimeError(f"No images found in {CALIB_IMAGES_DIR}. Did you run capture_calib_images.py?")

    objp = prepare_object_points()
    obj_points = []
    img_points = []

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001,
    )

    example_image = cv2.imread(image_paths[0])
    if example_image is None:
        raise RuntimeError(f"Cannot read example image {image_paths[0]}")
    gray_example = cv2.cvtColor(example_image, cv2.COLOR_BGR2GRAY)
    img_size = gray_example.shape[::-1]

    for path_str in image_paths:
        path = Path(path_str)
        found, corners = find_corners_in_image(path, criteria)
        if found and corners is not None:
            obj_points.append(objp)
            img_points.append(corners)

    if len(obj_points) < 5:
        raise RuntimeError("Not enough valid calibration images with detected corners. Need at least ~5.")

    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points,
        img_points,
        img_size,
        None,
        None,
    )

    total_error = 0
    total_points = 0
    for i in range(len(obj_points)):
        img_points_proj, _ = cv2.projectPoints(
            obj_points[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs
        )
        err = cv2.norm(img_points[i], img_points_proj, cv2.NORM_L2)
        total_error += err ** 2
        total_points += len(obj_points[i])
    mean_error = np.sqrt(total_error / total_points)

    np.savez(
        CAMERA_PARAMS_PATH,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        img_width=img_size[0],
        img_height=img_size[1],
        reprojection_error=mean_error,
        rms=ret,
    )

    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]

    print("[INFO] Calibration finished.")
    print(f"       Saved parameters to: {CAMERA_PARAMS_PATH}")
    print("       Camera matrix K:")
    print(camera_matrix)
    print(f"       Distortion coefficients: {dist_coeffs.ravel()}")
    print(f"       Mean reprojection error: {mean_error:.4f} pixels")
    print(f"       Focal length in pixels: fx={fx:.2f}, fy={fy:.2f}")
    print(f"       Principal point (cx, cy): ({cx:.2f}, {cy:.2f})")

if __name__ == "__main__":
    calibrate_camera()
