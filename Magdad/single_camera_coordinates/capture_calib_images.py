# capture_calib_images.py

import cv2
import os
from pathlib import Path
from datetime import datetime
from config import (
    CALIB_IMAGES_DIR,
    CAMERA_INDEX,
    FRAME_WIDTH,
    FRAME_HEIGHT,
)

def open_webcam(camera_index: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open webcam with index {camera_index}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    return cap

def save_frame(frame, directory: Path) -> str:
    directory.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"calib_{timestamp}.png"
    full_path = directory / filename
    cv2.imwrite(str(full_path), frame)
    return str(full_path)

def run_capture_loop():
    cap = open_webcam(CAMERA_INDEX)
    print("[INFO] Press SPACE to capture an image, 'q' to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to grab frame from webcam.")
                break

            cv2.putText(
                frame,
                "Press SPACE to capture, 'q' to quit.",
                (30, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("Calibration Capture", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord(' '):
                saved_path = save_frame(frame, CALIB_IMAGES_DIR)
                print(f"[INFO] Saved calibration image: {saved_path}")
            elif key == ord('q'):
                print("[INFO] Quitting capture loop.")
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run_capture_loop()
