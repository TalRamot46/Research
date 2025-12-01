# triangulate_point.py

from pathlib import Path
import numpy as np
import cv2
import sys

from config import STEREO_PARAMS_PATH

def load_stereo_params(path: Path | str):
    data = np.load(path)
    K1 = data["K1"]
    d1 = data["d1"]
    K2 = data["K2"]
    d2 = data["d2"]
    R = data["R"]
    T = data["T"]
    baseline_mm = float(data["baseline_mm"])
    img_width = int(data["img_width"])
    img_height = int(data["img_height"])
    rms = float(data["rms"])
    return K1, d1, K2, d2, R, T, baseline_mm, img_width, img_height, rms

def get_correspondence(img_left, img_right):
    pts_left = []
    pts_right = []

    def cb_left(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            pts_left.clear()
            pts_left.append((x, y))
            print(f"[INFO] Left point: ({x}, {y})")

    def cb_right(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            pts_right.clear()
            pts_right.append((x, y))
            print(f"[INFO] Right point: ({x}, {y})")

    cv2.namedWindow("Left")
    cv2.namedWindow("Right")
    cv2.setMouseCallback("Left", cb_left)
    cv2.setMouseCallback("Right", cb_right)

    while True:
        disp_left = img_left.copy()
        disp_right = img_right.copy()

        if pts_left:
            cv2.circle(disp_left, pts_left[0], 5, (0, 0, 255), -1)
        if pts_right:
            cv2.circle(disp_right, pts_right[0], 5, (0, 0, 255), -1)

        cv2.imshow("Left", disp_left)
        cv2.imshow("Right", disp_right)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        if key == ord(' '):
            if pts_left and pts_right:
                break

    cv2.destroyAllWindows()

    if not pts_left or not pts_right:
        raise RuntimeError("Need one point in each image (press SPACE when both are selected).")

    return pts_left[0], pts_right[0]

# take a picture using the two cameras and save the images to path called images
def take_picture():
    # taking two pictures continue me
    out_path = Path(__file__).resolve().parent.parent / "images"
    # Ensure the images directory itself exists (not its parent)
    out_path.mkdir(parents=True, exist_ok=True)

    # Use DirectShow backend on Windows for better compatibility when available
    if sys.platform.startswith("win"):
        cap1 = cv2.VideoCapture(1, cv2.CAP_DSHOW)
        cap2 = cv2.VideoCapture(2, cv2.CAP_DSHOW)
    else:
        cap1 = cv2.VideoCapture(1)
        cap2 = cv2.VideoCapture(2)

    if not cap1 or not cap1.isOpened():
        print("Error: Cannot open camera index 1.", file=sys.stderr)
        sys.exit(2)

    if not cap2 or not cap2.isOpened():
        print("Error: Cannot open camera index 2.", file=sys.stderr)
        sys.exit(2)
    

    # Warm up the cameras by reading a few frames (helps some webcams)
    for _ in range(3):
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    cap1.release()
    cap2.release()

    if not ret1 or frame1 is None or not ret2 or frame2 is None:
        print("Error: Failed to capture frame from cameras.", file=sys.stderr)
        sys.exit(3)

    # Save the image as JPEG
    if not cv2.imwrite(str(out_path / "left.jpg"), frame1):
        print(f"Error: Failed to write image to {out_path}", file=sys.stderr)
        sys.exit(4)
    
    if not cv2.imwrite(str(out_path / "right.jpg"), frame2):
        print(f"Error: Failed to write image to {out_path}", file=sys.stderr)
        sys.exit(4)

    print(f"Saved images to {out_path}")

def triangulate_3d(
    K1, d1, K2, d2, R, T,
    pt_left, pt_right,
):
    p1 = np.array([[pt_left[0], pt_left[1]]], dtype=np.float64)
    p2 = np.array([[pt_right[0], pt_right[1]]], dtype=np.float64)

    p1_norm = cv2.undistortPoints(p1, K1, d1)
    p2_norm = cv2.undistortPoints(p2, K2, d2)

    p1_norm = p1_norm.reshape(2, 1)
    p2_norm = p2_norm.reshape(2, 1)

    P1 = np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = np.hstack((R, T))

    X_hom = cv2.triangulatePoints(P1, P2, p1_norm, p2_norm)
    X = X_hom[:3] / X_hom[3]

    X = X.reshape(3)
    return X

def main():
    K1, d1, K2, d2, R, T, baseline_mm, img_w, img_h, rms = load_stereo_params(STEREO_PARAMS_PATH)
    print("[INFO] Loaded stereo calibration.")
    print(f"       RMS error: {rms:.4f}  |  baseline: {baseline_mm:.2f} mm")

    take_picture()
    left_path = Path(__file__).resolve().parent.parent / "images" / "left.jpg"
    right_path = Path(__file__).resolve().parent.parent / "images" / "right.jpg"
    # Use string paths for OpenCV I/O to avoid compatibility issues with Path objects
    img_left = cv2.imread(str(left_path))
    img_right = cv2.imread(str(right_path))
    if img_left is None or img_right is None:
        raise RuntimeError("Could not read one of the images.")

    if img_left.shape[:2] != img_right.shape[:2]:
        raise RuntimeError("Left/right images must have same resolution.")

    pt_left, pt_right = get_correspondence(img_left, img_right)

    X_mm = triangulate_3d(K1, d1, K2, d2, R, T, pt_left, pt_right)

    distance_mm = float(np.linalg.norm(X_mm))
    distance_m = distance_mm / 1000.0

    print("\n[RESULT] 3D point coordinates (left camera frame):")
    print(f"         X = {X_mm[0]:.2f} mm")
    print(f"         Y = {X_mm[1]:.2f} mm")
    print(f"         Z = {X_mm[2]:.2f} mm")
    print(f"         Distance from left camera: {distance_m:.3f} m")

if __name__ == "__main__":
    main()
