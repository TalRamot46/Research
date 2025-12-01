# config.py

from pathlib import Path

# === Calibration pattern parameters ===
CHESSBOARD_ROWS = 7
CHESSBOARD_COLS = 10
SQUARE_SIZE_MM = 24.0

# === File system paths ===
BASE_DIR = Path(__file__).resolve().parent

CALIB_IMAGES_DIR = BASE_DIR / "calib_images"
CALIB_IMAGES_DIR.mkdir(exist_ok=True)

# NEW: stereo calibration image dirs
CALIB_LEFT_DIR = BASE_DIR / "calib_left"
CALIB_RIGHT_DIR = BASE_DIR / "calib_right"
CALIB_LEFT_DIR.mkdir(exist_ok=True)
CALIB_RIGHT_DIR.mkdir(exist_ok=True)

# Single-camera intrinsics output
CAMERA_PARAMS_PATH = BASE_DIR / "camera_params.npz"

# NEW: stereo params output
STEREO_PARAMS_PATH = BASE_DIR / "stereo_params.npz"

# === Stereo camera parameters ===
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# === Sensor physical size (IMX179) ===
SENSOR_WIDTH_MM = 6.18
SENSOR_HEIGHT_MM = 5.85

# --- YOLO model ---
# Path to your trained YOLO weights (pendulum detector)
YOLO_MODEL_PATH = r"pendulum_yolo\\results\\pendulum_yolo73\weights\best.pt"  # change to your actual path

# --- Stereo live capture ---
CAMERA_LEFT_INDEX = 1   # make sure these match your real devices
CAMERA_RIGHT_INDEX = 2

# Optional: if you know the physical baseline very accurately (in mm),
# you can put it here to rescale the translation vector T when loading.
# If you don't care, just set BASELINE_REAL_MM = None
BASELINE_REAL_MM = None  # or e.g. 300.0
