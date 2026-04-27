import json
import time
from pathlib import Path

import cv2
import numpy as np

CALIBRATION_JSON_PATH = Path("camera_calibration.json")

CAMERA_SOURCE = 0
CAMERA_BACKEND = "auto"
TARGET_WIDTH = 1280
TARGET_HEIGHT = 720
TARGET_FPS = 30
TARGET_FOURCC = "MJPG"
WINDOW_TITLE = "ChArUco / ArUco Detection"

DICTIONARY_NAME = "DICT_APRILTAG_36h11"
CHARUCO_SQUARES_X = 7
CHARUCO_SQUARES_Y = 5
CHARUCO_SQUARE_LENGTH_MM = 40.0
CHARUCO_MARKER_LENGTH_MM = 30.0

# Used for standalone ArUco pose estimation.
# If you print another physical size, update this constant.
ARUCO_MARKER_LENGTH_MM = 168.0
AXIS_LENGTH_MM = 50.0

FPS_UPDATE_PERIOD_SEC = 0.5
TEXT_COLOR = (0, 255, 0)
CHARUCO_COLOR = (255, 200, 0)
ARUCO_COLOR = (0, 200, 255)

BACKEND_MAP = {
    "auto": cv2.CAP_ANY,
    "dshow": cv2.CAP_DSHOW,
    "v4l2": cv2.CAP_V4L2,
}


def load_calibration():
    payload = json.loads(CALIBRATION_JSON_PATH.read_text(encoding="utf-8"))
    camera_matrix = np.array(payload["camera_matrix"], dtype=np.float64)
    dist_coeffs = np.array(payload["dist_coeffs"], dtype=np.float64)
    return camera_matrix, dist_coeffs


def create_charuco_board():
    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(getattr(aruco, DICTIONARY_NAME))
    board = aruco.CharucoBoard(
        (CHARUCO_SQUARES_X, CHARUCO_SQUARES_Y),
        CHARUCO_SQUARE_LENGTH_MM,
        CHARUCO_MARKER_LENGTH_MM,
        dictionary,
    )
    return dictionary, board


def open_camera():
    params = [
        cv2.CAP_PROP_FOURCC,
        cv2.VideoWriter_fourcc(*TARGET_FOURCC),
        cv2.CAP_PROP_FRAME_WIDTH,
        TARGET_WIDTH,
        cv2.CAP_PROP_FRAME_HEIGHT,
        TARGET_HEIGHT,
        cv2.CAP_PROP_FPS,
        TARGET_FPS,
        cv2.CAP_PROP_BUFFERSIZE,
        1,
    ]
    backend = BACKEND_MAP.get(CAMERA_BACKEND, cv2.CAP_ANY)
    cap = cv2.VideoCapture(CAMERA_SOURCE, backend, params)
    if not cap.isOpened():
        return None

    for _ in range(10):
        cap.read()
    return cap


def build_marker_object_points(marker_length_mm):
    half = marker_length_mm * 0.5
    return np.array(
        [
            [-half, half, 0.0],
            [half, half, 0.0],
            [half, -half, 0.0],
            [-half, -half, 0.0],
        ],
        dtype=np.float32,
    )


def solve_marker_pose(marker_corners, camera_matrix, dist_coeffs, marker_length_mm):
    image_points = np.asarray(marker_corners, dtype=np.float32).reshape(4, 2)
    object_points = build_marker_object_points(marker_length_mm)
    ok, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    return ok, rvec, tvec


def solve_charuco_pose(board, charuco_corners, charuco_ids, camera_matrix, dist_coeffs):
    object_points, image_points = board.matchImagePoints(charuco_corners, charuco_ids)
    if object_points is None or image_points is None:
        return False, None, None
    if len(object_points) < 4 or len(image_points) < 4:
        return False, None, None

    ok, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    return ok, rvec, tvec


def pose_to_camera_coordinates_and_angles(rvec, tvec):
    rotation_matrix, _ = cv2.Rodrigues(rvec)
    camera_position = -rotation_matrix.T @ tvec

    sy = np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        roll = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        pitch = np.arctan2(-rotation_matrix[2, 0], sy)
        yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        roll = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        pitch = np.arctan2(-rotation_matrix[2, 0], sy)
        yaw = 0.0

    return camera_position.reshape(3), np.degrees([roll, pitch, yaw])


def draw_pose_text(frame, title, rvec, tvec, origin_x, origin_y, color):
    camera_position, camera_angles_deg = pose_to_camera_coordinates_and_angles(rvec, tvec)
    lines = [
        title,
        f"Cam XYZ mm: {camera_position[0]:7.1f} {camera_position[1]:7.1f} {camera_position[2]:7.1f}",
        f"Cam RPY deg: {camera_angles_deg[0]:6.1f} {camera_angles_deg[1]:6.1f} {camera_angles_deg[2]:6.1f}",
    ]

    for index, line in enumerate(lines):
        y = origin_y + index * 28
        cv2.putText(
            frame,
            line,
            (origin_x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )


def main():
    camera_matrix, dist_coeffs = load_calibration()
    dictionary, board = create_charuco_board()
    detector = cv2.aruco.CharucoDetector(board)
    cap = open_camera()
    if cap is None:
        print(f"Error: cannot open camera source {CAMERA_SOURCE}")
        return

    current_fps = 0.0
    fps_frame_count = 0
    fps_started_at = time.perf_counter()

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Frame read error")
            break

        fps_frame_count += 1
        now = time.perf_counter()
        elapsed = now - fps_started_at
        if elapsed >= FPS_UPDATE_PERIOD_SEC:
            current_fps = fps_frame_count / elapsed
            fps_frame_count = 0
            fps_started_at = now

        display = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(gray)

        if marker_ids is not None and len(marker_ids) > 0:
            cv2.aruco.drawDetectedMarkers(display, marker_corners, marker_ids, ARUCO_COLOR)
            for corners, marker_id in zip(marker_corners, marker_ids.flatten()):
                pose_ok, rvec, tvec = solve_marker_pose(
                    corners,
                    camera_matrix,
                    dist_coeffs,
                    ARUCO_MARKER_LENGTH_MM,
                )
                if pose_ok:
                    cv2.drawFrameAxes(
                        display,
                        camera_matrix,
                        dist_coeffs,
                        rvec,
                        tvec,
                        AXIS_LENGTH_MM,
                        2,
                    )
                    anchor = np.asarray(corners, dtype=np.float32).reshape(4, 2)[0].astype(int)
                    cv2.putText(
                        display,
                        f"id={int(marker_id)}",
                        (int(anchor[0]), int(anchor[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        ARUCO_COLOR,
                        2,
                        cv2.LINE_AA,
                    )
                    draw_pose_text(
                        display,
                        f"ArUco id={int(marker_id)}",
                        rvec,
                        tvec,
                        10,
                        95,
                        ARUCO_COLOR,
                    )

        if charuco_ids is not None and len(charuco_ids) >= 4:
            cv2.aruco.drawDetectedCornersCharuco(display, charuco_corners, charuco_ids, CHARUCO_COLOR)
            pose_ok, rvec, tvec = solve_charuco_pose(
                board,
                charuco_corners,
                charuco_ids,
                camera_matrix,
                dist_coeffs,
            )
            if pose_ok:
                cv2.drawFrameAxes(
                    display,
                    camera_matrix,
                    dist_coeffs,
                    rvec,
                    tvec,
                    AXIS_LENGTH_MM * 1.5,
                    3,
                )
                cv2.putText(
                    display,
                    f"ChArUco corners: {len(charuco_ids)}",
                    (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    CHARUCO_COLOR,
                    2,
                    cv2.LINE_AA,
                )
                draw_pose_text(
                    display,
                    "ChArUco pose",
                    rvec,
                    tvec,
                    10,
                    95,
                    CHARUCO_COLOR,
                )

        cv2.putText(
            display,
            f"FPS: {current_fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            TEXT_COLOR,
            2,
            cv2.LINE_AA,
        )

        cv2.imshow(WINDOW_TITLE, display)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
