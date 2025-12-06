# distance_from_size.py

from pathlib import Path
import numpy as np
import cv2

from config import (
    CAMERA_PARAMS_PATH,
    SENSOR_WIDTH_MM,
    SENSOR_HEIGHT_MM,
)

def load_camera_params(path: Path | str):
    data = np.load(path)
    camera_matrix = data["camera_matrix"]
    dist_coeffs = data["dist_coeffs"]
    img_width = int(data["img_width"])
    img_height = int(data["img_height"])
    reprojection_error = float(data["reprojection_error"])
    return camera_matrix, dist_coeffs, img_width, img_height, reprojection_error

def compute_focal_length_mm(camera_matrix: np.ndarray, img_width: int, img_height: int) -> float:
    fx_pixels = camera_matrix[0, 0]
    fy_pixels = camera_matrix[1, 1]

    fx_mm = fx_pixels * (SENSOR_WIDTH_MM / img_width)
    fy_mm = fy_pixels * (SENSOR_HEIGHT_MM / img_height)
    f_mm = 0.5 * (fx_mm + fy_mm)
    return f_mm

def estimate_distance_from_vertical_size(
    f_mm: float,
    real_height_mm: float,
    object_height_pixels: float,
    img_height_pixels: int,
) -> float:
    projected_height_mm = object_height_pixels * (SENSOR_HEIGHT_MM / img_height_pixels)
    distance_mm = (f_mm * real_height_mm) / projected_height_mm
    return distance_mm

def measure_object_height_pixels(image_path: str) -> float:
    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError(f"Cannot read image {image_path}")

    h, w = img.shape[:2]
    print(f"[INFO] Image size: {w} x {h}")
    print("[INFO] Click top and bottom of the object in the image.")
    print("[INFO] Press any key when done, 'r' to reset.")

    points = []

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            print(f"[INFO] Point selected: ({x}, {y})")

    clone = img.copy()
    cv2.namedWindow("Measure Object")
    cv2.setMouseCallback("Measure Object", mouse_callback)

    while True:
        img_display = clone.copy()
        for p in points:
            cv2.circle(img_display, p, 5, (0, 0, 255), -1)
        if len(points) == 2:
            cv2.line(img_display, points[0], points[1], (0, 255, 0), 2)

        cv2.imshow("Measure Object", img_display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('r'):
            points.clear()
            print("[INFO] Points reset.")
        elif key != 255 and key != ord('r'):
            break

    cv2.destroyAllWindows()

    if len(points) != 2:
        raise RuntimeError("Need exactly two points (top and bottom of the object).")

    (x1, y1), (x2, y2) = points
    height_pixels = abs(y2 - y1)
    print(f"[INFO] Measured object height: {height_pixels:.2f} pixels")
    return float(height_pixels)

def main():
    camera_matrix, dist_coeffs, img_width, img_height, reprojection_error = load_camera_params(CAMERA_PARAMS_PATH)
    print("[INFO] Loaded camera parameters.")
    print("       Camera matrix:")
    print(camera_matrix)
    print(f"       Reprojection error: {reprojection_error:.4f} pixels")

    f_mm = compute_focal_length_mm(camera_matrix, img_width, img_height)
    print(f"[INFO] Effective focal length: {f_mm:.3f} mm")

    image_path = input("Enter path to an image of the object: ").strip()
    real_height_mm = float(input("Enter real object height in mm: ").strip())

    object_height_pixels = measure_object_height_pixels(image_path)

    distance_mm = estimate_distance_from_vertical_size(
        f_mm=f_mm,
        real_height_mm=real_height_mm,
        object_height_pixels=object_height_pixels,
        img_height_pixels=img_height,
    )

    distance_m = distance_mm / 1000.0
    print(f"[RESULT] Estimated distance to object: {distance_m:.3f} m")

if __name__ == "__main__":
    main()
