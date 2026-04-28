import json
from pathlib import Path

import cv2
import numpy as np

IMAGE_DIR = Path("images")
IMAGE_GLOB = "*.jpg"

# Current photos contain a classic chessboard with 15x10 inner corners.
PATTERN_TYPE = "chessboard"
CHESSBOARD_INNER_CORNERS = (15, 10)
SQUARE_SIZE_MM = 50.0

SHOW_DETECTIONS = False
SAVE_DEBUG_OVERLAYS = True
DEBUG_DIR = Path("debug_corners")

NPZ_OUTPUT_PATH = Path("camera_calibration.npz")
JSON_OUTPUT_PATH = Path("camera_calibration.json")
YAML_OUTPUT_PATH = Path("camera_calibration.yaml")

SUBPIX_WINDOW = (11, 11)
SUBPIX_ZERO_ZONE = (-1, -1)
SUBPIX_CRITERIA = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    50,
    1e-4,
)

CALIBRATION_FLAGS = 0


def build_chessboard_object_points():
    cols, rows = CHESSBOARD_INNER_CORNERS
    objp = np.zeros((rows * cols, 3), np.float32)
    grid = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp[:, :2] = grid * SQUARE_SIZE_MM
    return objp


def collect_image_points(image_paths):
    object_template = build_chessboard_object_points()
    object_points = []
    image_points = []
    image_size = None
    used_images = []

    if SAVE_DEBUG_OVERLAYS:
        DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Skip unreadable image: {image_path}")
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_size = gray.shape[::-1]

        found, corners = cv2.findChessboardCornersSB(
            gray,
            CHESSBOARD_INNER_CORNERS,
            flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY,
        )
        if not found:
            print(f"Pattern not found: {image_path.name}")
            continue

        corners = cv2.cornerSubPix(
            gray,
            corners,
            SUBPIX_WINDOW,
            SUBPIX_ZERO_ZONE,
            SUBPIX_CRITERIA,
        )

        object_points.append(object_template.copy())
        image_points.append(corners)
        used_images.append(image_path)

        overlay = image.copy()
        cv2.drawChessboardCorners(
            overlay,
            CHESSBOARD_INNER_CORNERS,
            corners,
            found,
        )

        if SAVE_DEBUG_OVERLAYS:
            debug_path = DEBUG_DIR / image_path.name
            cv2.imwrite(str(debug_path), overlay)

        if SHOW_DETECTIONS:
            cv2.imshow("Calibration detections", overlay)
            cv2.waitKey(250)

        print(f"Detected pattern: {image_path.name}")

    if SHOW_DETECTIONS:
        cv2.destroyAllWindows()

    return object_points, image_points, image_size, used_images


def compute_reprojection_error(object_points, image_points, rvecs, tvecs, camera_matrix, dist_coeffs):
    total_error = 0.0
    total_points = 0
    per_image_errors = []

    for objp, imgp, rvec, tvec in zip(object_points, image_points, rvecs, tvecs):
        projected, _ = cv2.projectPoints(objp, rvec, tvec, camera_matrix, dist_coeffs)
        error = cv2.norm(imgp, projected, cv2.NORM_L2)
        points_count = len(projected)
        per_image_errors.append(error / points_count)
        total_error += error * error
        total_points += points_count

    rms = np.sqrt(total_error / total_points) if total_points else float("nan")
    return float(rms), [float(v) for v in per_image_errors]


def save_results(camera_matrix, dist_coeffs, rms, reprojection_error, image_size, used_images, rvecs, tvecs):
    np.savez(
        NPZ_OUTPUT_PATH,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        rms=rms,
        reprojection_error=reprojection_error,
        image_width=image_size[0],
        image_height=image_size[1],
        pattern_type=PATTERN_TYPE,
        chessboard_inner_corners=np.array(CHESSBOARD_INNER_CORNERS, dtype=np.int32),
        square_size_mm=SQUARE_SIZE_MM,
        used_images=np.array([str(path) for path in used_images]),
        rvecs=np.array(rvecs, dtype=np.float64),
        tvecs=np.array(tvecs, dtype=np.float64),
    )

    payload = {
        "pattern_type": PATTERN_TYPE,
        "chessboard_inner_corners": list(CHESSBOARD_INNER_CORNERS),
        "square_size_mm": SQUARE_SIZE_MM,
        "image_size": {
            "width": int(image_size[0]),
            "height": int(image_size[1]),
        },
        "rms": float(rms),
        "reprojection_error": float(reprojection_error),
        "camera_matrix": camera_matrix.tolist(),
        "dist_coeffs": dist_coeffs.tolist(),
        "used_images": [str(path) for path in used_images],
    }
    JSON_OUTPUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    yaml_lines = [
        "image_width: {}".format(int(image_size[0])),
        "image_height: {}".format(int(image_size[1])),
        "camera_name: usb_camera",
        "camera_matrix:",
        "  rows: 3",
        "  cols: 3",
        "  data: [{}]".format(", ".join(f"{value:.12f}" for value in camera_matrix.reshape(-1))),
        "distortion_model: plumb_bob",
        "distortion_coefficients:",
        f"  rows: {dist_coeffs.shape[0]}",
        f"  cols: {dist_coeffs.shape[1]}",
        "  data: [{}]".format(", ".join(f"{value:.12f}" for value in dist_coeffs.reshape(-1))),
        f"rms: {float(rms):.12f}",
        f"reprojection_error: {float(reprojection_error):.12f}",
        f"pattern_type: {PATTERN_TYPE}",
        "chessboard_inner_corners: [{}, {}]".format(*CHESSBOARD_INNER_CORNERS),
        f"square_size_mm: {float(SQUARE_SIZE_MM):.6f}",
    ]
    YAML_OUTPUT_PATH.write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")


def calibrate_camera():
    image_paths = sorted(IMAGE_DIR.glob(IMAGE_GLOB))
    if not image_paths:
        raise RuntimeError(f"No calibration images found in {IMAGE_DIR.resolve()}")

    object_points, image_points, image_size, used_images = collect_image_points(image_paths)
    if not object_points or image_size is None:
        raise RuntimeError("Calibration pattern was not detected in any image.")

    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        object_points,
        image_points,
        image_size,
        None,
        None,
        flags=CALIBRATION_FLAGS,
    )

    reprojection_error, per_image_errors = compute_reprojection_error(
        object_points,
        image_points,
        rvecs,
        tvecs,
        camera_matrix,
        dist_coeffs,
    )

    save_results(
        camera_matrix,
        dist_coeffs,
        rms,
        reprojection_error,
        image_size,
        used_images,
        rvecs,
        tvecs,
    )

    print()
    print("Calibration completed")
    print(f"  images used         : {len(used_images)} / {len(image_paths)}")
    print(f"  image size          : {image_size[0]}x{image_size[1]}")
    print(f"  RMS                 : {rms:.6f}")
    print(f"  reprojection error  : {reprojection_error:.6f} px")
    print(f"  npz                 : {NPZ_OUTPUT_PATH.resolve()}")
    print(f"  json                : {JSON_OUTPUT_PATH.resolve()}")
    print(f"  yaml                : {YAML_OUTPUT_PATH.resolve()}")
    print("  camera matrix:")
    print(camera_matrix)
    print("  dist coeffs:")
    print(dist_coeffs.ravel())
    print("  per-image errors:")
    for image_path, error in zip(used_images, per_image_errors):
        print(f"    {image_path.name}: {error:.6f} px")


def main():
    if PATTERN_TYPE != "chessboard":
        raise RuntimeError("This script is currently configured only for chessboard calibration.")
    calibrate_camera()


if __name__ == "__main__":
    main()
