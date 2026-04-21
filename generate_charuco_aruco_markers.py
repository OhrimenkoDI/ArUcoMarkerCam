import json
from pathlib import Path

import cv2
import numpy as np

CALIBRATION_JSON_PATH = Path("camera_calibration.json")
OUTPUT_DIR = Path("markers")

DICTIONARY_NAME = "DICT_APRILTAG_36h11"

CHARUCO_SQUARES_X = 7
CHARUCO_SQUARES_Y = 5
CHARUCO_SQUARE_LENGTH_MM = 40.0
CHARUCO_MARKER_LENGTH_MM = 30.0
CHARUCO_MARGIN_PX = 80

PIXELS_PER_MM = 12.0
STANDALONE_MARKER_IDS = [0, 1, 2, 3, 4, 5]
STANDALONE_MARKER_SIZES_MM = [180.0]
MARKER_BORDER_BITS = 1

METADATA_PATH = OUTPUT_DIR / "marker_pack.json"
PREVIEW_PATH = OUTPUT_DIR / "marker_pack_preview.png"


def load_calibration():
    payload = json.loads(CALIBRATION_JSON_PATH.read_text(encoding="utf-8"))
    camera_matrix = np.array(payload["camera_matrix"], dtype=np.float64)
    dist_coeffs = np.array(payload["dist_coeffs"], dtype=np.float64)
    return payload, camera_matrix, dist_coeffs


def get_dictionary():
    aruco = cv2.aruco
    dictionary_id = getattr(aruco, DICTIONARY_NAME)
    return aruco.getPredefinedDictionary(dictionary_id)


def generate_charuco_board(dictionary):
    aruco = cv2.aruco
    board = aruco.CharucoBoard(
        (CHARUCO_SQUARES_X, CHARUCO_SQUARES_Y),
        CHARUCO_SQUARE_LENGTH_MM,
        CHARUCO_MARKER_LENGTH_MM,
        dictionary,
    )

    board_width_px = int(round(CHARUCO_SQUARES_X * CHARUCO_SQUARE_LENGTH_MM * PIXELS_PER_MM))
    board_height_px = int(round(CHARUCO_SQUARES_Y * CHARUCO_SQUARE_LENGTH_MM * PIXELS_PER_MM))
    board_image = board.generateImage(
        (board_width_px, board_height_px),
        marginSize=CHARUCO_MARGIN_PX,
        borderBits=MARKER_BORDER_BITS,
    )

    board_path = OUTPUT_DIR / "charuco_board.png"
    cv2.imwrite(str(board_path), board_image)

    return {
        "path": str(board_path),
        "board_size_px": [board_width_px, board_height_px],
        "board_size_mm": [
            CHARUCO_SQUARES_X * CHARUCO_SQUARE_LENGTH_MM,
            CHARUCO_SQUARES_Y * CHARUCO_SQUARE_LENGTH_MM,
        ],
        "squares_x": CHARUCO_SQUARES_X,
        "squares_y": CHARUCO_SQUARES_Y,
        "square_length_mm": CHARUCO_SQUARE_LENGTH_MM,
        "marker_length_mm": CHARUCO_MARKER_LENGTH_MM,
    }


def generate_standalone_markers(dictionary):
    marker_records = []
    singles_dir = OUTPUT_DIR / "aruco_single"
    singles_dir.mkdir(parents=True, exist_ok=True)

    for marker_id in STANDALONE_MARKER_IDS:
        for marker_size_mm in STANDALONE_MARKER_SIZES_MM:
            side_px = int(round(marker_size_mm * PIXELS_PER_MM))
            marker_image = cv2.aruco.generateImageMarker(
                dictionary,
                marker_id,
                side_px,
                borderBits=MARKER_BORDER_BITS,
            )
            filename = f"aruco_id{marker_id:02d}_{int(round(marker_size_mm)):03d}mm.png"
            marker_path = singles_dir / filename
            cv2.imwrite(str(marker_path), marker_image)
            marker_records.append(
                {
                    "id": marker_id,
                    "size_mm": marker_size_mm,
                    "size_px": side_px,
                    "path": str(marker_path),
                }
            )

    return marker_records


def estimate_marker_pixel_size(camera_matrix, marker_size_mm, distance_mm):
    fx = float(camera_matrix[0, 0])
    return fx * marker_size_mm / distance_mm


def build_preview(marker_records, charuco_record):
    preview_width = 2200
    preview_height = 1600
    canvas = np.full((preview_height, preview_width, 3), 255, dtype=np.uint8)

    title = "Marker Pack Preview"
    subtitle = "ChArUco board + standalone ArUco markers"
    cv2.putText(canvas, title, (60, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(canvas, subtitle, (60, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 40, 40), 2, cv2.LINE_AA)

    board_image = cv2.imread(charuco_record["path"], cv2.IMREAD_GRAYSCALE)
    board_bgr = cv2.cvtColor(board_image, cv2.COLOR_GRAY2BGR)
    board_target_width = 1100
    scale = board_target_width / board_bgr.shape[1]
    board_target_height = int(round(board_bgr.shape[0] * scale))
    board_resized = cv2.resize(board_bgr, (board_target_width, board_target_height), interpolation=cv2.INTER_AREA)
    canvas[180:180 + board_target_height, 60:60 + board_target_width] = board_resized
    cv2.rectangle(canvas, (60, 180), (60 + board_target_width, 180 + board_target_height), (0, 0, 0), 2)

    cv2.putText(
        canvas,
        f"ChArUco {charuco_record['squares_x']}x{charuco_record['squares_y']}  "
        f"square={charuco_record['square_length_mm']:.0f} mm  marker={charuco_record['marker_length_mm']:.0f} mm",
        (60, 180 + board_target_height + 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )

    x0 = 1280
    y0 = 180
    dx = 260
    dy = 260
    thumb_size = 180
    for index, record in enumerate(marker_records[:6]):
        marker = cv2.imread(record["path"], cv2.IMREAD_GRAYSCALE)
        marker = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
        marker = cv2.resize(marker, (thumb_size, thumb_size), interpolation=cv2.INTER_NEAREST)
        row = index // 2
        col = index % 2
        x = x0 + col * dx
        y = y0 + row * dy
        canvas[y:y + thumb_size, x:x + thumb_size] = marker
        cv2.rectangle(canvas, (x, y), (x + thumb_size, y + thumb_size), (0, 0, 0), 2)
        cv2.putText(
            canvas,
            f"id={record['id']}  {record['size_mm']:.0f} mm",
            (x, y + thumb_size + 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

    cv2.imwrite(str(PREVIEW_PATH), canvas)


def save_metadata(calibration_payload, camera_matrix, dist_coeffs, charuco_record, marker_records):
    recommended_distances = {}
    for marker_size_mm in STANDALONE_MARKER_SIZES_MM:
        recommended_distances[f"{int(round(marker_size_mm))}mm"] = {
            "approx_pixels_at_500mm": estimate_marker_pixel_size(camera_matrix, marker_size_mm, 500.0),
            "approx_pixels_at_1000mm": estimate_marker_pixel_size(camera_matrix, marker_size_mm, 1000.0),
            "approx_pixels_at_2000mm": estimate_marker_pixel_size(camera_matrix, marker_size_mm, 2000.0),
        }

    payload = {
        "dictionary_name": DICTIONARY_NAME,
        "pixels_per_mm": PIXELS_PER_MM,
        "calibration_file": str(CALIBRATION_JSON_PATH),
        "camera_calibration": {
            "image_size": calibration_payload["image_size"],
            "rms": calibration_payload["rms"],
            "reprojection_error": calibration_payload["reprojection_error"],
            "camera_matrix": camera_matrix.tolist(),
            "dist_coeffs": dist_coeffs.tolist(),
        },
        "charuco_board": charuco_record,
        "standalone_markers": marker_records,
        "detection_hints": {
            "preferred_marker_family": "ChArUco for board pose, standalone ArUco for single-tag tracking",
            "recommended_distances": recommended_distances,
            "min_practical_marker_pixels": 40,
        },
        "preview_image": str(PREVIEW_PATH),
    }
    METADATA_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    calibration_payload, camera_matrix, dist_coeffs = load_calibration()
    dictionary = get_dictionary()

    charuco_record = generate_charuco_board(dictionary)
    marker_records = generate_standalone_markers(dictionary)
    build_preview(marker_records, charuco_record)
    save_metadata(calibration_payload, camera_matrix, dist_coeffs, charuco_record, marker_records)

    print("Marker pack generated")
    print(f"  output dir : {OUTPUT_DIR.resolve()}")
    print(f"  board      : {(OUTPUT_DIR / 'charuco_board.png').resolve()}")
    print(f"  preview    : {PREVIEW_PATH.resolve()}")
    print(f"  metadata   : {METADATA_PATH.resolve()}")


if __name__ == "__main__":
    main()
