import json
import time
from pathlib import Path

import cv2
import numpy as np

CALIBRATION_JSON_PATH = Path("camera_calibration.json")
MARKER_LAYOUT_JSON_PATH = Path("marker_layout.json")

CAMERA_SOURCE = 0
CAMERA_BACKEND = "auto"
TARGET_WIDTH = 1280
TARGET_HEIGHT = 720
TARGET_FPS = 30
TARGET_FOURCC = "MJPG"

FPS_UPDATE_PERIOD_SEC = 0.5
CONSOLE_PRINT_PERIOD_SEC = 0.5

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


def load_marker_layout():
    payload = json.loads(MARKER_LAYOUT_JSON_PATH.read_text(encoding="utf-8"))
    dictionary_name = payload["dictionary_name"]
    marker_length_mm = float(payload["marker_length_mm"])

    marker_world_transforms = {}
    for marker_id_text, marker_payload in payload["markers"].items():
        marker_id = int(marker_id_text)
        rvec = np.array(marker_payload["rvec"], dtype=np.float64).reshape(3, 1)
        tvec = np.array(marker_payload["tvec"], dtype=np.float64).reshape(3, 1)
        marker_world_transforms[marker_id] = rt_to_transform(rvec, tvec)

    return payload, dictionary_name, marker_length_mm, marker_world_transforms


def get_dictionary(dictionary_name):
    aruco = cv2.aruco
    return aruco.getPredefinedDictionary(getattr(aruco, dictionary_name))


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
    if not ok:
        return False, None, None, float("inf")

    projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs)
    reprojection_error = float(
        np.mean(np.linalg.norm(image_points - projected.reshape(4, 2), axis=1))
    )
    return True, rvec, tvec, reprojection_error


def rt_to_transform(rvec, tvec):
    rotation_matrix, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation_matrix
    transform[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
    return transform


def invert_transform(transform):
    inverse = np.eye(4, dtype=np.float64)
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -(rotation.T @ translation)
    return inverse


def rotation_matrix_to_euler_deg(rotation_matrix):
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

    return np.degrees([roll, pitch, yaw])


def average_transforms(transforms):
    if not transforms:
        return None

    translations = np.array([transform[:3, 3] for transform in transforms], dtype=np.float64)
    rotvecs = []
    for transform in transforms:
        rvec, _ = cv2.Rodrigues(transform[:3, :3])
        rotvecs.append(rvec.reshape(3))

    mean_transform = np.eye(4, dtype=np.float64)
    mean_transform[:3, 3] = np.mean(translations, axis=0)
    mean_rvec = np.mean(np.array(rotvecs, dtype=np.float64), axis=0).reshape(3, 1)
    mean_transform[:3, :3], _ = cv2.Rodrigues(mean_rvec)
    return mean_transform


def compute_pose_consistency_mm(candidates):
    if len(candidates) < 2:
        return None
    positions = np.array([transform[:3, 3] for transform in candidates], dtype=np.float64)
    return float(np.mean(np.std(positions, axis=0)))


def detect_marker_poses(frame, dictionary, camera_matrix, dist_coeffs, marker_length_mm):
    detector = cv2.aruco.ArucoDetector(dictionary)
    marker_corners, marker_ids, _ = detector.detectMarkers(frame)
    poses = {}

    if marker_ids is None or len(marker_ids) == 0:
        return poses

    for corners, marker_id in zip(marker_corners, marker_ids.flatten()):
        pose_ok, rvec, tvec, reprojection_error = solve_marker_pose(
            corners,
            camera_matrix,
            dist_coeffs,
            marker_length_mm,
        )
        if pose_ok:
            poses[int(marker_id)] = {
                "rvec": rvec,
                "tvec": tvec,
                "camera_from_marker": rt_to_transform(rvec, tvec),
                "reprojection_error": reprojection_error,
            }

    return poses


def estimate_world_from_camera(marker_world_transforms, detected_poses):
    world_from_camera_candidates = []
    used_marker_ids = []
    for marker_id, pose in detected_poses.items():
        if marker_id not in marker_world_transforms:
            continue
        world_from_marker = marker_world_transforms[marker_id]
        world_from_camera_candidates.append(
            world_from_marker @ invert_transform(pose["camera_from_marker"])
        )
        used_marker_ids.append(marker_id)

    if not world_from_camera_candidates:
        return None, [], None

    consistency_mm = compute_pose_consistency_mm(world_from_camera_candidates)
    world_from_camera = average_transforms(world_from_camera_candidates)
    return world_from_camera, used_marker_ids, consistency_mm


def format_pose_line(world_from_camera, used_marker_ids, consistency_mm, fps, detected_poses):
    xyz_mm = world_from_camera[:3, 3]
    rpy_deg = rotation_matrix_to_euler_deg(world_from_camera[:3, :3])
    marker_list = ", ".join(str(marker_id) for marker_id in sorted(set(used_marker_ids)))
    reprojection = [
        detected_poses[marker_id]["reprojection_error"]
        for marker_id in used_marker_ids
        if marker_id in detected_poses
    ]
    avg_reprojection = float(np.mean(reprojection)) if reprojection else float("nan")

    consistency_text = "---" if consistency_mm is None else f"{consistency_mm:.1f}"
    return (
        f"FPS={fps:5.1f} | markers=[{marker_list}] | "
        f"XYZ_mm=({xyz_mm[0]:8.1f}, {xyz_mm[1]:8.1f}, {xyz_mm[2]:8.1f}) | "
        f"RPY_deg=({rpy_deg[0]:7.2f}, {rpy_deg[1]:7.2f}, {rpy_deg[2]:7.2f}) | "
        f"consistency_mm={consistency_text} | reproj_px={avg_reprojection:.2f}"
    )


def main():
    if not CALIBRATION_JSON_PATH.exists():
        print(f"Error: calibration file not found: {CALIBRATION_JSON_PATH}")
        return
    if not MARKER_LAYOUT_JSON_PATH.exists():
        print(f"Error: marker layout file not found: {MARKER_LAYOUT_JSON_PATH}")
        return

    camera_matrix, dist_coeffs = load_calibration()
    layout_payload, dictionary_name, marker_length_mm, marker_world_transforms = load_marker_layout()
    dictionary = get_dictionary(dictionary_name)
    cap = open_camera()
    if cap is None:
        print(f"Error: cannot open camera source {CAMERA_SOURCE}")
        return

    print(
        f"Started detection. Base marker: {layout_payload['base_marker_id']}, "
        f"map markers: {sorted(marker_world_transforms)}"
    )
    print("Press Ctrl+C in console to stop.")

    current_fps = 0.0
    fps_frame_count = 0
    fps_started_at = time.perf_counter()
    last_console_print_at = 0.0
    last_status = None

    try:
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

            detected_poses = detect_marker_poses(
                frame,
                dictionary,
                camera_matrix,
                dist_coeffs,
                marker_length_mm,
            )
            world_from_camera, used_marker_ids, consistency_mm = estimate_world_from_camera(
                marker_world_transforms,
                detected_poses,
            )

            should_print = (now - last_console_print_at) >= CONSOLE_PRINT_PERIOD_SEC

            if world_from_camera is not None:
                line = format_pose_line(
                    world_from_camera,
                    used_marker_ids,
                    consistency_mm,
                    current_fps,
                    detected_poses,
                )
                if should_print or line != last_status:
                    print(line)
                    last_console_print_at = now
                    last_status = line
            else:
                visible_ids = sorted(detected_poses)
                status = (
                    "Known markers not found in frame."
                    if not visible_ids
                    else f"Detected markers not in map: {visible_ids}"
                )
                if should_print or status != last_status:
                    print(status)
                    last_console_print_at = now
                    last_status = status
    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        cap.release()


if __name__ == "__main__":
    main()
