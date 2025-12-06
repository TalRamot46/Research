# yolo_stereo_live.py

import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from config import (
    STEREO_PARAMS_PATH,
    YOLO_MODEL_PATH,
    CAMERA_LEFT_INDEX,
    CAMERA_RIGHT_INDEX,
    FRAME_WIDTH,
    FRAME_HEIGHT,
    BASELINE_REAL_MM,
)


# ---------- Stereo calibration loading & triangulation ----------

def load_stereo_params(path: str | Path):
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

    return K1, d1, K2, d2, R, T, baseline_mm, img_width, img_height, rms


def undistort_pixel_points(pixels: np.ndarray, K: np.ndarray, d: np.ndarray) -> np.ndarray:
    pts = pixels.reshape(-1, 1, 2).astype(np.float64)
    pts_norm = cv2.undistortPoints(pts, K, d)
    return pts_norm.reshape(-1, 2).T  # shape (2, N)


def triangulate_point(
    K1: np.ndarray,
    d1: np.ndarray,
    K2: np.ndarray,
    d2: np.ndarray,
    R: np.ndarray,
    T: np.ndarray,
    pt_left: tuple[float, float],
    pt_right: tuple[float, float],
) -> np.ndarray | None:
    p1 = np.array([[pt_left[0], pt_left[1]]], dtype=np.float64)
    p2 = np.array([[pt_right[0], pt_right[1]]], dtype=np.float64)

    p1_norm = undistort_pixel_points(p1, K1, d1)
    p2_norm = undistort_pixel_points(p2, K2, d2)

    P1 = np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = np.hstack((R, T))

    X_hom = cv2.triangulatePoints(P1, P2, p1_norm, p2_norm)
    if X_hom[3, 0] == 0:
        return None

    X = X_hom[:3, 0] / X_hom[3, 0]
    return X  # in mm


# ---------- YOLO detection helpers ----------

def load_yolo_model(weights_path: str | Path) -> YOLO:
    model = YOLO(str(weights_path))
    return model


def detect_pendulum_center(
    frame_bgr: np.ndarray,
    model: YOLO,
    pendulum_class_id: int = 0,
) -> tuple[tuple[int, int] | None, tuple[int, int, int, int] | None]:
    results = model(frame_bgr, verbose=False)[0]

    best_box = None
    best_conf = -1.0

    if results.boxes is None:
        return None, None

    for box in results.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])

        if cls != pendulum_class_id:
            continue

        if conf > best_conf:
            xyxy = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = xyxy.astype(int)
            best_box = (x1, y1, x2, y2)
            best_conf = conf

    if best_box is None:
        return None, None

    x1, y1, x2, y2 = best_box
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    center = (cx, cy)

    return center, best_box


# ---------- Camera opening & display ----------

def open_camera(idx: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(idx)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {idx}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    return cap


def draw_detection(
    frame: np.ndarray,
    box: tuple[int, int, int, int] | None,
    center: tuple[int, int] | None,
    text: str | None = None,
):
    if box is not None:
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    if center is not None:
        cv2.circle(frame, center, 4, (0, 0, 255), -1)
    if text is not None:
        org = (10, 30)
        cv2.putText(
            frame,
            text,
            org,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )


# ---------- Main loop ----------

def main():
    print("[INFO] Loading stereo parameters...")
    K1, d1, K2, d2, R, T, baseline_mm, img_w, img_h, rms = load_stereo_params(STEREO_PARAMS_PATH)
    print(f"[INFO] Stereo RMS error: {rms:.4f}, baseline: {baseline_mm:.1f} mm")

    print("[INFO] Loading YOLO model...")
    model = load_yolo_model(YOLO_MODEL_PATH)
    print("[INFO] Model loaded.")

    cap_left = open_camera(CAMERA_LEFT_INDEX)
    cap_right = open_camera(CAMERA_RIGHT_INDEX)

    print("[INFO] Press 'q' to quit.")

    try:
        while True:
            ok_l, frame_l = cap_left.read()
            ok_r, frame_r = cap_right.read()
            if not ok_l or not ok_r:
                print("[ERROR] Failed to grab frames.")
                break

            center_l, box_l = detect_pendulum_center(frame_l, model)
            center_r, box_r = detect_pendulum_center(frame_r, model)

            distance_m = None
            if center_l is not None and center_r is not None:
                X_mm = triangulate_point(K1, d1, K2, d2, R, T, center_l, center_r)
                if X_mm is not None and X_mm[2] > 0:
                    distance_mm = float(np.linalg.norm(X_mm))
                    distance_m = distance_mm / 1000.0

            if distance_m is not None:
                text_l = f"Dist: {distance_m:.2f} m"
                text_r = f"Dist: {distance_m:.2f} m"
            else:
                text_l = "No stereo match"
                text_r = "No stereo match"

            draw_detection(frame_l, box_l, center_l, text_l)
            draw_detection(frame_r, box_r, center_r, text_r)

            concat = cv2.hconcat([frame_l, frame_r])
            cv2.imshow("YOLO + Stereo Distance (Left | Right)", concat)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    finally:
        cap_left.release()
        cap_right.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
