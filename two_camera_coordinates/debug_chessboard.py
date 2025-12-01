# debug_chessboard.py

import cv2
from pathlib import Path
from config import (
    CHESSBOARD_ROWS, 
    CHESSBOARD_COLS
    )

def main():
    img_path = input("Enter path to one LEFT image (or RIGHT): ").strip()
    if not img_path:
        print("No path given.")
        return

    img = cv2.imread(img_path)
    if img is None:
        print(f"Could not read {img_path}")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    pattern_size = (CHESSBOARD_COLS, CHESSBOARD_ROWS)

    flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH
        + cv2.CALIB_CB_NORMALIZE_IMAGE
        + cv2.CALIB_CB_FAST_CHECK
    )

    ret, corners = cv2.findChessboardCorners(gray, pattern_size, flags)

    print(f"findChessboardCorners returned: {ret}")
    if ret:
        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            30,
            0.001,
        )
        corners = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1), criteria
        )
        cv2.drawChessboardCorners(img, pattern_size, corners, ret)

    cv2.imshow("Chessboard Debug", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
