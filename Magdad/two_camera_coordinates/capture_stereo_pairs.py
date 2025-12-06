# capture_stereo_pairs.py

import cv2
from pathlib import Path
from datetime import datetime
from config import (
    CALIB_LEFT_DIR,
    CALIB_RIGHT_DIR,
    CAMERA_LEFT_INDEX,
    CAMERA_RIGHT_INDEX,
    FRAME_WIDTH,
    FRAME_HEIGHT,
)

def open_camera(index: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {index}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    return cap

def ensure_dirs():
    CALIB_LEFT_DIR.mkdir(exist_ok=True)
    CALIB_RIGHT_DIR.mkdir(exist_ok=True)

def build_pair_filenames() -> tuple[str, str]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    left_name = f"left_{timestamp}.png"
    right_name = f"right_{timestamp}.png"
    left_path = CALIB_LEFT_DIR / left_name
    right_path = CALIB_RIGHT_DIR / right_name
    return str(left_path), str(right_path)

def run_capture_loop():
    ensure_dirs()
    cap_left = open_camera(CAMERA_LEFT_INDEX)
    cap_right = open_camera(CAMERA_RIGHT_INDEX)

    print("[INFO] Press SPACE to capture stereo pair, 'q' to quit.")

    try:
        while True:
            ret_l, frame_l = cap_left.read()
            ret_r, frame_r = cap_right.read()

            if not ret_l or not ret_r:
                print("[ERROR] Failed to grab frames from both cameras.")
                break

            concat = cv2.hconcat([frame_l, frame_r])
            cv2.putText(
                concat,
                "SPACE: capture pair  |  q: quit",
                (30, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("Stereo Capture (Left | Right)", concat)
            key = cv2.waitKey(1) & 0xFF

            if key == ord(' '):
                left_path, right_path = build_pair_filenames()
                cv2.imwrite(left_path, frame_l)
                cv2.imwrite(right_path, frame_r)
                print(f"[INFO] Saved pair:\n       {left_path}\n       {right_path}")
            elif key == ord('q'):
                print("[INFO] Quitting capture.")
                break
    finally:
        cap_left.release()
        cap_right.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run_capture_loop()
