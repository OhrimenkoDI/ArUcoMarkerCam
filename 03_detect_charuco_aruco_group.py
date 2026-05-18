import json
import socket
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

CALIBRATION_JSON_PATH = Path("camera_calibration.json")
MARKER_LAYOUT_JSON_PATH = Path("marker_layout.json")
VERIFICATION_LOG_DIR = Path("logs")

CAMERA_SOURCE = 0
CAMERA_BACKEND = "auto"
TARGET_WIDTH = 1280
TARGET_HEIGHT = 720
TARGET_FPS = 30
TARGET_FOURCC = "MJPG"
WINDOW_TITLE = "Group ArUco Detection"
SELECTION_WINDOW_TITLE = "Select Mode"
GRAPH_WINDOW_TITLE = "Mode 3 pose history"

DICTIONARY_NAME = "DICT_APRILTAG_36h11"
ARUCO_MARKER_LENGTH_MM = 285.0
AXIS_LENGTH_MM = 60.0
BASE_MARKER_ID = 0
LEARNING_MODE_3D = 1
LEARNING_MODE_PLANAR = 2
VERIFICATION_MODE = 3
UDP_MONITOR_MODE = 4
UDP_LOG_LISTEN_IP = "0.0.0.0"
UDP_LOG_PORT = 15050

FPS_UPDATE_PERIOD_SEC = 0.5
TEXT_COLOR = (0, 255, 0)
INFO_COLOR = (255, 220, 0)
MARKER_COLOR = (0, 200, 255)
BASE_MARKER_COLOR = (0, 255, 0)
ERROR_COLOR = (80, 80, 255)
QUALITY_GOOD_COLOR = (0, 220, 0)
QUALITY_MED_COLOR = (0, 200, 220)
QUALITY_BAD_COLOR = (60, 60, 255)

GRAPH_HISTORY_FRAMES = 300
GRAPH_WIDTH = 1280
GRAPH_HEIGHT = 720
GRAPH_BG_COLOR = (24, 24, 24)
GRAPH_GRID_COLOR = (55, 55, 55)
GRAPH_TEXT_COLOR = (225, 225, 225)
ARTIFACT_JUMP_MM = 1000.0
MAX_REPROJECTION_ERROR_PX = 3.0
HARD_MAX_REPROJECTION_ERROR_PX = 5.0
ROBUST_CANDIDATE_RESIDUAL_MM = 500.0
SINGLE_MARKER_MAX_JUMP_MM = 500.0
DUAL_MARKER_MAX_SPREAD_MM = 250.0
EMA_EXCLUSION_THRESHOLD_PX = 6.0
LONG_POSE_LOSS_FRAMES = 30

# EMA smoothing factor for per-marker reprojection error and quality score
EMA_ALPHA = 0.1

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


def get_dictionary():
    aruco = cv2.aruco
    return aruco.getPredefinedDictionary(getattr(aruco, DICTIONARY_NAME))


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
    if ok:
        projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs)
        reprojection_error = float(
            np.mean(np.linalg.norm(image_points - projected.reshape(4, 2), axis=1))
        )
    else:
        reprojection_error = float("inf")
    return ok, rvec, tvec, reprojection_error


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


def transform_to_rt(transform):
    rvec, _ = cv2.Rodrigues(transform[:3, :3])
    tvec = transform[:3, 3].reshape(3, 1)
    return rvec, tvec


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


def transform_to_log_pose(transform):
    if transform is None:
        return None
    position = transform[:3, 3]
    angles_deg = rotation_matrix_to_euler_deg(transform[:3, :3])
    return {
        "xyz_mm": [float(value) for value in position],
        "rpy_deg": [float(value) for value in angles_deg],
    }


def vector_to_log(values):
    return [float(value) for value in np.asarray(values, dtype=np.float64).reshape(-1)]


def create_verification_log_file():
    VERIFICATION_LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    path = VERIFICATION_LOG_DIR / f"mode3_verification_{timestamp}.jsonl"
    log_file = path.open("w", encoding="utf-8")
    print(f"Mode 3 log: {path.resolve()}")
    return path, log_file


def rotation_matrix_to_quaternion(rotation_matrix):
    matrix = np.asarray(rotation_matrix, dtype=np.float64)
    trace = np.trace(matrix)
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (matrix[2, 1] - matrix[1, 2]) / s
        y = (matrix[0, 2] - matrix[2, 0]) / s
        z = (matrix[1, 0] - matrix[0, 1]) / s
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        s = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
        w = (matrix[2, 1] - matrix[1, 2]) / s
        x = 0.25 * s
        y = (matrix[0, 1] + matrix[1, 0]) / s
        z = (matrix[0, 2] + matrix[2, 0]) / s
    elif matrix[1, 1] > matrix[2, 2]:
        s = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
        w = (matrix[0, 2] - matrix[2, 0]) / s
        x = (matrix[0, 1] + matrix[1, 0]) / s
        y = 0.25 * s
        z = (matrix[1, 2] + matrix[2, 1]) / s
    else:
        s = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
        w = (matrix[1, 0] - matrix[0, 1]) / s
        x = (matrix[0, 2] + matrix[2, 0]) / s
        y = (matrix[1, 2] + matrix[2, 1]) / s
        z = 0.25 * s
    quat = np.array([w, x, y, z], dtype=np.float64)
    return quat / np.linalg.norm(quat)


def quaternion_to_rotation_matrix(quaternion):
    w, x, y, z = np.asarray(quaternion, dtype=np.float64)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def average_transforms(transforms):
    if not transforms:
        return None

    translations = np.array([transform[:3, 3] for transform in transforms], dtype=np.float64)
    quaternions = []
    reference_quaternion = None
    for transform in transforms:
        quaternion = rotation_matrix_to_quaternion(transform[:3, :3])
        if reference_quaternion is None:
            reference_quaternion = quaternion
        elif np.dot(quaternion, reference_quaternion) < 0.0:
            quaternion = -quaternion
        quaternions.append(quaternion)

    mean_transform = np.eye(4, dtype=np.float64)
    mean_transform[:3, 3] = np.mean(translations, axis=0)
    mean_quaternion = np.mean(np.array(quaternions, dtype=np.float64), axis=0)
    mean_quaternion /= np.linalg.norm(mean_quaternion)
    mean_transform[:3, :3] = quaternion_to_rotation_matrix(mean_quaternion)
    return mean_transform


def blend_transforms(old_transform, new_transform, alpha):
    t = (1.0 - alpha) * old_transform[:3, 3] + alpha * new_transform[:3, 3]
    q1 = rotation_matrix_to_quaternion(old_transform[:3, :3])
    q2 = rotation_matrix_to_quaternion(new_transform[:3, :3])
    if np.dot(q1, q2) < 0.0:
        q2 = -q2
    dot = float(np.clip(np.dot(q1, q2), -1.0, 1.0))
    if dot > 0.9995:
        q = q1 + alpha * (q2 - q1)
    else:
        theta = np.arccos(dot)
        q = (np.sin((1.0 - alpha) * theta) * q1 + np.sin(alpha * theta) * q2) / np.sin(theta)
    q /= np.linalg.norm(q)
    result = np.eye(4, dtype=np.float64)
    result[:3, 3] = t
    result[:3, :3] = quaternion_to_rotation_matrix(q)
    return result


def compute_pose_consistency(candidates):
    if len(candidates) < 2:
        return None, None
    positions = np.array([t[:3, 3] for t in candidates], dtype=np.float64)
    trans_mm = float(np.mean(np.std(positions, axis=0)))
    quaternions = []
    ref_q = None
    for t in candidates:
        q = rotation_matrix_to_quaternion(t[:3, :3])
        if ref_q is None:
            ref_q = q
        elif np.dot(q, ref_q) < 0.0:
            q = -q
        quaternions.append(q)
    mean_q = np.mean(quaternions, axis=0)
    mean_q /= np.linalg.norm(mean_q)
    rot_deg = float(np.mean([
        np.degrees(2.0 * np.arccos(min(1.0, abs(float(np.dot(q, mean_q))))))
        for q in quaternions
    ]))
    return trans_mm, rot_deg


def update_error_ema(ema_dict, marker_id, new_error):
    if marker_id in ema_dict:
        ema_dict[marker_id] = EMA_ALPHA * new_error + (1.0 - EMA_ALPHA) * ema_dict[marker_id]
    else:
        ema_dict[marker_id] = new_error


def parse_mode_arg():
    for arg in sys.argv[1:]:
        value = arg.strip().lower()
        if value in {"mode=1", "1", "--mode=1"}:
            return LEARNING_MODE_3D
        if value in {"mode=2", "2", "--mode=2"}:
            return LEARNING_MODE_PLANAR
        if value in {"mode=3", "3", "--mode=3"}:
            return VERIFICATION_MODE
        if value in {"mode=4", "4", "--mode=4"}:
            return UDP_MONITOR_MODE
    return None


def select_mode_by_key():
    canvas = np.full((420, 980, 3), 30, dtype=np.uint8)
    lines = [
        "Select mode",
        "Press 1  - learning mode (3D map)",
        "Press 2  - learning mode (planar floor map)",
        "Press 3  - verification mode",
        "Press 4  - UDP pose monitor",
        f"Base marker id: {BASE_MARKER_ID}",
        "Esc - exit",
    ]
    for index, line in enumerate(lines):
        scale = 1.1 if index == 0 else 0.9
        y = 70 + index * 55
        cv2.putText(
            canvas,
            line,
            (40, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (240, 240, 240),
            2,
            cv2.LINE_AA,
        )

    cv2.imshow(SELECTION_WINDOW_TITLE, canvas)
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord("1"):
            cv2.destroyWindow(SELECTION_WINDOW_TITLE)
            return LEARNING_MODE_3D
        if key == ord("2"):
            cv2.destroyWindow(SELECTION_WINDOW_TITLE)
            return LEARNING_MODE_PLANAR
        if key == ord("3"):
            cv2.destroyWindow(SELECTION_WINDOW_TITLE)
            return VERIFICATION_MODE
        if key == ord("4"):
            cv2.destroyWindow(SELECTION_WINDOW_TITLE)
            return UDP_MONITOR_MODE
        if key == 27:
            cv2.destroyWindow(SELECTION_WINDOW_TITLE)
            return None


def choose_mode():
    mode = parse_mode_arg()
    if mode is not None:
        return mode
    return select_mode_by_key()


def detect_marker_poses(
    frame,
    dictionary,
    camera_matrix,
    dist_coeffs,
    marker_length_mm=ARUCO_MARKER_LENGTH_MM,
):
    detector = cv2.aruco.ArucoDetector(dictionary)
    marker_corners, marker_ids, _ = detector.detectMarkers(frame)
    poses = {}

    if marker_ids is None or len(marker_ids) == 0:
        return marker_corners, marker_ids, poses

    for corners, marker_id in zip(marker_corners, marker_ids.flatten()):
        pose_ok, rvec, tvec, reprojection_error = solve_marker_pose(
            corners,
            camera_matrix,
            dist_coeffs,
            marker_length_mm,
        )
        if pose_ok:
            poses[int(marker_id)] = {
                "corners": corners,
                "rvec": rvec,
                "tvec": tvec,
                "camera_from_marker": rt_to_transform(rvec, tvec),
                "reprojection_error": reprojection_error,
            }

    return marker_corners, marker_ids, poses


def load_marker_layout():
    payload = json.loads(MARKER_LAYOUT_JSON_PATH.read_text(encoding="utf-8"))
    marker_transforms = {}
    marker_counts = {}
    for marker_id_text, marker_payload in payload["markers"].items():
        marker_id = int(marker_id_text)
        rvec = np.array(marker_payload["rvec"], dtype=np.float64).reshape(3, 1)
        tvec = np.array(marker_payload["tvec"], dtype=np.float64).reshape(3, 1)
        marker_transforms[marker_id] = rt_to_transform(rvec, tvec)
        marker_counts[marker_id] = int(marker_payload.get("observation_count", 0))
    return payload, marker_transforms, marker_counts


def planarize_transform(transform):
    planar = np.array(transform, dtype=np.float64, copy=True)
    planar[:3, 3][2] = 0.0
    yaw_deg = float(rotation_matrix_to_euler_deg(planar[:3, :3])[2])
    yaw_rad = np.radians(yaw_deg)
    cos_yaw = float(np.cos(yaw_rad))
    sin_yaw = float(np.sin(yaw_rad))
    planar[:3, :3] = np.array(
        [
            [cos_yaw, -sin_yaw, 0.0],
            [sin_yaw, cos_yaw, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return planar


def normalize_marker_world_estimates(marker_world_estimates, planar=False):
    if BASE_MARKER_ID not in marker_world_estimates:
        raise RuntimeError(f"Base marker {BASE_MARKER_ID} is missing from learned map.")

    base_transform = marker_world_estimates[BASE_MARKER_ID]["transform"]
    normalization_transform = invert_transform(base_transform)

    normalized = {}
    for marker_id, marker_data in marker_world_estimates.items():
        normalized_transform = normalization_transform @ marker_data["transform"]
        if planar:
            normalized_transform = planarize_transform(normalized_transform)
        normalized[marker_id] = {
            "transform": normalized_transform,
            "count": marker_data["count"],
        }

    normalized[BASE_MARKER_ID]["transform"] = np.eye(4, dtype=np.float64)
    return normalized


def save_marker_layout(marker_world_estimates, planar=False):
    normalized_estimates = normalize_marker_world_estimates(marker_world_estimates, planar=planar)
    markers_payload = {}
    for marker_id in sorted(normalized_estimates):
        transform = normalized_estimates[marker_id]["transform"]
        rvec, tvec = transform_to_rt(transform)
        markers_payload[str(marker_id)] = {
            "rvec": rvec.reshape(3).tolist(),
            "tvec": tvec.reshape(3).tolist(),
            "observation_count": int(normalized_estimates[marker_id]["count"]),
            "xyz_mm": transform[:3, 3].tolist(),
            "rpy_deg": rotation_matrix_to_euler_deg(transform[:3, :3]).tolist(),
        }

    payload = {
        "base_marker_id": BASE_MARKER_ID,
        "marker_length_mm": ARUCO_MARKER_LENGTH_MM,
        "dictionary_name": DICTIONARY_NAME,
        "layout_mode": "planar" if planar else "3d",
        "markers": markers_payload,
    }
    MARKER_LAYOUT_JSON_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def update_marker_world_estimates(
    marker_world_estimates,
    detected_poses,
    planar=False,
    previous_world_from_camera=None,
    marker_error_ema=None,
    frames_since_previous_pose=None,
):
    marker_world_transforms = {
        marker_id: data["transform"]
        for marker_id, data in marker_world_estimates.items()
    }
    world_from_camera, used_marker_ids, (trans_mm, rot_deg), candidate_details, diagnostics = (
        estimate_world_from_camera_robust(
            marker_world_transforms,
            detected_poses,
            previous_world_from_camera,
            marker_error_ema=marker_error_ema,
            frames_since_previous_pose=frames_since_previous_pose,
        )
    )
    diagnostics["skipped_marker_updates"] = []

    if world_from_camera is None:
        return None, [], (None, None), diagnostics

    # Для обновления карты предпочитаем базовый маркер, если он прошел тот же
    # робастный фильтр. Это уменьшает риск, что ошибочный небазовый маркер сам
    # подтянет свою позицию в карте.
    world_from_camera_trusted = world_from_camera
    accepted_ids = set(diagnostics["accepted_marker_ids"])
    if BASE_MARKER_ID in accepted_ids:
        for detail in candidate_details:
            if int(detail["marker_id"]) == BASE_MARKER_ID:
                world_from_camera_trusted = detail["world_from_camera"]
                break

    new_marker_ids = []
    for marker_id, pose in detected_poses.items():
        reprojection_error = float(pose["reprojection_error"])
        if reprojection_error > MAX_REPROJECTION_ERROR_PX:
            diagnostics["skipped_marker_updates"].append(
                {
                    "marker_id": int(marker_id),
                    "reason": "reprojection_error",
                    "reprojection_error_px": reprojection_error,
                }
            )
            continue

        estimated_world_from_marker = world_from_camera_trusted @ pose["camera_from_marker"]
        if planar:
            estimated_world_from_marker = planarize_transform(estimated_world_from_marker)

        if marker_id in marker_world_estimates:
            count = marker_world_estimates[marker_id]["count"]
            if marker_id != BASE_MARKER_ID:
                old_transform = marker_world_estimates[marker_id]["transform"]
                marker_jump_mm = float(
                    np.linalg.norm(estimated_world_from_marker[:3, 3] - old_transform[:3, 3])
                )
                if marker_jump_mm > ROBUST_CANDIDATE_RESIDUAL_MM:
                    diagnostics["skipped_marker_updates"].append(
                        {
                            "marker_id": int(marker_id),
                            "reason": "map_position_jump",
                            "jump_mm": marker_jump_mm,
                        }
                    )
                    continue
                marker_world_estimates[marker_id]["count"] = count + 1
                alpha = max(0.02, 1.0 / (count + 1))
                updated = blend_transforms(old_transform, estimated_world_from_marker, alpha)
                if planar:
                    updated = planarize_transform(updated)
                marker_world_estimates[marker_id]["transform"] = updated
            else:
                marker_world_estimates[marker_id]["count"] = count + 1
        else:
            marker_world_estimates[marker_id] = {
                "transform": estimated_world_from_marker,
                "count": 1,
            }
            new_marker_ids.append(marker_id)

    return world_from_camera, new_marker_ids, (trans_mm, rot_deg), diagnostics


def build_world_from_camera_candidates(marker_world_transforms, detected_poses):
    world_from_camera_candidates = []
    used_marker_ids = []
    candidate_details = []
    for marker_id, pose in detected_poses.items():
        if marker_id not in marker_world_transforms:
            continue
        world_from_marker = marker_world_transforms[marker_id]
        candidate = world_from_marker @ invert_transform(pose["camera_from_marker"])
        world_from_camera_candidates.append(candidate)
        used_marker_ids.append(marker_id)
        candidate_details.append(
            {
                "marker_id": marker_id,
                "world_from_camera": candidate,
                "world_from_marker": world_from_marker,
                "camera_from_marker": pose["camera_from_marker"],
                "rvec": pose["rvec"],
                "tvec": pose["tvec"],
                "reprojection_error": pose["reprojection_error"],
                "corners": pose["corners"],
            }
        )

    return world_from_camera_candidates, used_marker_ids, candidate_details


def estimate_world_from_camera(marker_world_transforms, detected_poses):
    world_from_camera_candidates, used_marker_ids, _ = build_world_from_camera_candidates(
        marker_world_transforms,
        detected_poses,
    )

    if not world_from_camera_candidates:
        return None, [], (None, None)

    trans_mm, rot_deg = compute_pose_consistency(world_from_camera_candidates)
    world_from_camera = average_transforms(world_from_camera_candidates)
    return world_from_camera, used_marker_ids, (trans_mm, rot_deg)


def estimate_world_from_camera_with_diagnostics(marker_world_transforms, detected_poses):
    world_from_camera_candidates, used_marker_ids, candidate_details = build_world_from_camera_candidates(
        marker_world_transforms,
        detected_poses,
    )

    if not world_from_camera_candidates:
        return None, [], (None, None), []

    trans_mm, rot_deg = compute_pose_consistency(world_from_camera_candidates)
    world_from_camera = average_transforms(world_from_camera_candidates)
    return world_from_camera, used_marker_ids, (trans_mm, rot_deg), candidate_details


def estimate_world_from_camera_robust(
    marker_world_transforms,
    detected_poses,
    previous_world_from_camera=None,
    marker_error_ema=None,
    frames_since_previous_pose=None,
):
    ema_excluded_ids = []
    if marker_error_ema:
        clean_poses = {}
        for mid, pose in detected_poses.items():
            if marker_error_ema.get(mid, 0.0) > EMA_EXCLUSION_THRESHOLD_PX:
                ema_excluded_ids.append(int(mid))
            else:
                clean_poses[mid] = pose
        detected_poses = clean_poses

    _, _, all_candidate_details = build_world_from_camera_candidates(
        marker_world_transforms,
        detected_poses,
    )

    quality_candidates = [
        detail
        for detail in all_candidate_details
        if float(detail["reprojection_error"]) <= MAX_REPROJECTION_ERROR_PX
    ]
    hard_rejected_ids = [
        int(detail["marker_id"])
        for detail in all_candidate_details
        if float(detail["reprojection_error"]) > HARD_MAX_REPROJECTION_ERROR_PX
    ]
    soft_rejected_ids = [
        int(detail["marker_id"])
        for detail in all_candidate_details
        if MAX_REPROJECTION_ERROR_PX < float(detail["reprojection_error"]) <= HARD_MAX_REPROJECTION_ERROR_PX
    ]

    def _diag(**overrides):
        base = {
            "input_candidate_count": len(all_candidate_details),
            "accepted_marker_ids": [],
            "ema_excluded_marker_ids": ema_excluded_ids,
            "soft_rejected_by_reprojection_error": soft_rejected_ids,
            "hard_rejected_by_reprojection_error": hard_rejected_ids,
            "rejected_by_residual": [],
            "rejected_by_spread": [],
            "rejected_single_marker_jump": [],
            "effective_jump_limit_mm": None,
            "reason": "ok",
        }
        base.update(overrides)
        return base

    if not quality_candidates:
        return None, [], (None, None), all_candidate_details, _diag(reason="no_quality_candidates")

    rejected_by_residual = []
    accepted_details = quality_candidates
    if len(quality_candidates) >= 3:
        candidate_positions = np.array(
            [detail["world_from_camera"][:3, 3] for detail in quality_candidates],
            dtype=np.float64,
        )
        median_position = np.median(candidate_positions, axis=0)
        filtered_details = []
        for detail in quality_candidates:
            residual = float(
                np.linalg.norm(detail["world_from_camera"][:3, 3] - median_position)
            )
            if residual <= ROBUST_CANDIDATE_RESIDUAL_MM:
                filtered_details.append(detail)
            else:
                rejected_by_residual.append(
                    {
                        "marker_id": int(detail["marker_id"]),
                        "residual_mm": residual,
                    }
                )
        if filtered_details:
            accepted_details = filtered_details
        else:
            closest_index = int(np.argmin(np.linalg.norm(candidate_positions - median_position, axis=1)))
            accepted_details = [quality_candidates[closest_index]]

    rejected_by_spread = []
    if len(accepted_details) == 2:
        t_spread, _ = compute_pose_consistency(
            [d["world_from_camera"] for d in accepted_details]
        )
        if t_spread is not None and t_spread > DUAL_MARKER_MAX_SPREAD_MM:
            better = min(accepted_details, key=lambda d: float(d["reprojection_error"]))
            worse = next(d for d in accepted_details if d is not better)
            rejected_by_spread.append({
                "marker_id": int(worse["marker_id"]),
                "spread_mm": float(t_spread),
            })
            accepted_details = [better]

    rejected_single_marker_jump = []
    effective_jump_limit = None
    if len(accepted_details) <= 2 and previous_world_from_camera is not None:
        avg_candidate = average_transforms([d["world_from_camera"] for d in accepted_details])
        jump_xy = float(np.linalg.norm(
            avg_candidate[:3, 3][:2] - previous_world_from_camera[:3, 3][:2]
        ))
        long_loss = (
            frames_since_previous_pose is not None
            and frames_since_previous_pose > LONG_POSE_LOSS_FRAMES
        )
        effective_jump_limit = SINGLE_MARKER_MAX_JUMP_MM * (4.0 if long_loss else 1.0)
        if jump_xy > effective_jump_limit:
            for d in accepted_details:
                rejected_single_marker_jump.append({
                    "marker_id": int(d["marker_id"]),
                    "jump_xy_mm": jump_xy,
                })
            return None, [], (None, None), all_candidate_details, _diag(
                rejected_by_residual=rejected_by_residual,
                rejected_by_spread=rejected_by_spread,
                rejected_single_marker_jump=rejected_single_marker_jump,
                effective_jump_limit_mm=effective_jump_limit,
                reason="single_marker_jump",
            )

    accepted_transforms = [detail["world_from_camera"] for detail in accepted_details]
    trans_mm, rot_deg = compute_pose_consistency(accepted_transforms)
    world_from_camera = average_transforms(accepted_transforms)
    used_marker_ids = [int(detail["marker_id"]) for detail in accepted_details]
    return world_from_camera, used_marker_ids, (trans_mm, rot_deg), all_candidate_details, _diag(
        accepted_marker_ids=used_marker_ids,
        rejected_by_residual=rejected_by_residual,
        rejected_by_spread=rejected_by_spread,
        effective_jump_limit_mm=effective_jump_limit,
    )


def draw_marker_visuals(
    frame, marker_corners, marker_ids, detected_poses, camera_matrix, dist_coeffs,
    marker_counts=None, marker_error_ema=None,
):
    if marker_ids is None or len(marker_ids) == 0:
        return

    cv2.aruco.drawDetectedMarkers(frame, marker_corners, marker_ids, MARKER_COLOR)
    for marker_id in marker_ids.flatten():
        marker_id = int(marker_id)
        if marker_id not in detected_poses:
            continue

        pose = detected_poses[marker_id]
        color = BASE_MARKER_COLOR if marker_id == BASE_MARKER_ID else MARKER_COLOR
        cv2.drawFrameAxes(
            frame,
            camera_matrix,
            dist_coeffs,
            pose["rvec"],
            pose["tvec"],
            AXIS_LENGTH_MM,
            2,
        )

        corners_px = np.asarray(pose["corners"], dtype=np.float32).reshape(4, 2)
        anchor = corners_px[0].astype(int)
        top_y = int(corners_px[:, 1].min())

        count_text = ""
        if marker_counts is not None and marker_id in marker_counts:
            count_text = f" n={marker_counts[marker_id]}"

        error_text = ""
        if marker_error_ema is not None and marker_id in marker_error_ema:
            error_text = f" e={marker_error_ema[marker_id]:.1f}px"

        cv2.putText(
            frame,
            f"id={marker_id}{count_text}",
            (anchor[0], top_y - 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA,
        )
        if error_text:
            cv2.putText(
                frame,
                error_text.strip(),
                (anchor[0], top_y - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                ERROR_COLOR,
                2,
                cv2.LINE_AA,
            )


def draw_multiline_text(frame, lines, x, y, color):
    for index, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (x, y + index * 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )


def draw_world_pose_text(frame, title, world_from_camera, x, y, color):
    position = world_from_camera[:3, 3]
    angles_deg = rotation_matrix_to_euler_deg(world_from_camera[:3, :3])
    lines = [
        title,
        f"World XYZ mm: {position[0]:7.1f} {position[1]:7.1f} {position[2]:7.1f}",
        f"World RPY deg: {angles_deg[0]:6.1f} {angles_deg[1]:6.1f} {angles_deg[2]:6.1f}",
    ]
    draw_multiline_text(frame, lines, x, y, color)


def draw_quality_score(frame, trans_ema, rot_ema, x, y):
    if trans_ema is None:
        label = "Map quality: --- (need 2+ markers)"
        color = INFO_COLOR
    elif trans_ema < 5.0 and rot_ema < 0.5:
        label = f"Map quality: {trans_ema:.1f} mm  {rot_ema:.2f} deg  GOOD"
        color = QUALITY_GOOD_COLOR
    elif trans_ema < 20.0 and rot_ema < 2.0:
        label = f"Map quality: {trans_ema:.1f} mm  {rot_ema:.2f} deg  OK"
        color = QUALITY_MED_COLOR
    else:
        label = f"Map quality: {trans_ema:.1f} mm  {rot_ema:.2f} deg  POOR"
        color = QUALITY_BAD_COLOR
    cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)


def build_verification_graph_sample(world_from_camera, used_marker_ids, trans_mm, rot_deg, detected_poses):
    sample = {
        "x": np.nan,
        "y": np.nan,
        "z": np.nan,
        "roll": np.nan,
        "pitch": np.nan,
        "yaw": np.nan,
        "trans": np.nan if trans_mm is None else float(trans_mm),
        "rot": np.nan if rot_deg is None else float(rot_deg),
        "err": np.nan,
        "markers": float(len(set(used_marker_ids))),
    }

    if world_from_camera is not None:
        position = world_from_camera[:3, 3]
        angles_deg = rotation_matrix_to_euler_deg(world_from_camera[:3, :3])
        sample.update({
            "x": float(position[0]),
            "y": float(position[1]),
            "z": float(position[2]),
            "roll": float(angles_deg[0]),
            "pitch": float(angles_deg[1]),
            "yaw": float(angles_deg[2]),
        })

    visible_errors = [
        float(pose["reprojection_error"])
        for pose in detected_poses.values()
        if np.isfinite(pose["reprojection_error"])
    ]
    if visible_errors:
        sample["err"] = float(np.mean(visible_errors))

    return sample


def build_udp_graph_sample(packet):
    sample = {
        "x": np.nan,
        "y": np.nan,
        "z": np.nan,
        "roll": np.nan,
        "pitch": np.nan,
        "yaw": np.nan,
        "trans": np.nan if packet.get("consistency_mm") is None else float(packet["consistency_mm"]),
        "rot": np.nan if packet.get("consistency_deg") is None else float(packet["consistency_deg"]),
        "err": np.nan,
        "markers": float(len(set(packet.get("used_marker_ids") or []))),
    }

    pose = packet.get("pose")
    if pose is not None:
        xyz = pose.get("xyz_mm") or [np.nan, np.nan, np.nan]
        rpy = pose.get("rpy_deg") or [np.nan, np.nan, np.nan]
        sample.update(
            {
                "x": float(xyz[0]),
                "y": float(xyz[1]),
                "z": float(xyz[2]),
                "roll": float(rpy[0]),
                "pitch": float(rpy[1]),
                "yaw": float(rpy[2]),
            }
        )

    errors = [
        float(item["reprojection_error_px"])
        for item in packet.get("per_marker", [])
        if np.isfinite(float(item.get("reprojection_error_px", np.nan)))
    ]
    if errors:
        sample["err"] = float(np.mean(errors))

    return sample


def build_verification_log_record(
    frame_index,
    timestamp_sec,
    current_fps,
    layout_mode,
    marker_ids,
    detected_poses,
    marker_error_ema,
    world_from_camera,
    previous_world_from_camera,
    used_marker_ids,
    trans_mm,
    rot_deg,
    candidate_details,
    robust_diagnostics,
    previous_pose_frame,
):
    visible_marker_ids = []
    if marker_ids is not None:
        visible_marker_ids = [int(value) for value in marker_ids.flatten()]

    solved_marker_ids = sorted(int(value) for value in detected_poses)
    used_marker_set = {int(value) for value in used_marker_ids}
    unused_solved_ids = [value for value in solved_marker_ids if value not in used_marker_set]

    world_pose = transform_to_log_pose(world_from_camera)
    previous_delta = None
    artifact_flags = []
    frames_since_previous_pose = None
    if previous_pose_frame is not None:
        frames_since_previous_pose = int(frame_index - previous_pose_frame)
    if world_from_camera is not None and previous_world_from_camera is not None:
        delta_xyz = world_from_camera[:3, 3] - previous_world_from_camera[:3, 3]
        delta_xy = float(np.linalg.norm(delta_xyz[:2]))
        delta_xyz_norm = float(np.linalg.norm(delta_xyz))
        previous_delta = {
            "dxyz_mm": [float(value) for value in delta_xyz],
            "dxy_mm": delta_xy,
            "dxyz_norm_mm": delta_xyz_norm,
        }
        if delta_xy > ARTIFACT_JUMP_MM:
            artifact_flags.append("world_xy_jump_gt_1000mm")

    per_marker = []
    for detail in candidate_details:
        marker_id = int(detail["marker_id"])
        candidate = detail["world_from_camera"]
        candidate_pose = transform_to_log_pose(candidate)
        residual_xyz = None
        residual_norm = None
        if world_from_camera is not None:
            residual = candidate[:3, 3] - world_from_camera[:3, 3]
            residual_xyz = [float(value) for value in residual]
            residual_norm = float(np.linalg.norm(residual))

        corners_px = np.asarray(detail["corners"], dtype=np.float64).reshape(4, 2)
        per_marker.append(
            {
                "id": marker_id,
                "reprojection_error_px": float(detail["reprojection_error"]),
                "reprojection_error_ema_px": (
                    float(marker_error_ema[marker_id])
                    if marker_id in marker_error_ema
                    else None
                ),
                "camera_from_marker": transform_to_log_pose(detail["camera_from_marker"]),
                "world_from_marker_map": transform_to_log_pose(detail["world_from_marker"]),
                "world_from_camera_candidate": candidate_pose,
                "candidate_minus_average_xyz_mm": residual_xyz,
                "candidate_minus_average_norm_mm": residual_norm,
                "solvepnp_rvec": vector_to_log(detail["rvec"]),
                "solvepnp_tvec_mm": vector_to_log(detail["tvec"]),
                "corners_px": [[float(x), float(y)] for x, y in corners_px],
            }
        )

    return {
        "record_type": "frame",
        "frame": int(frame_index),
        "time_sec": float(timestamp_sec),
        "wall_time_unix": float(time.time()),
        "fps": float(current_fps),
        "layout_mode": layout_mode,
        "artifact_flags": artifact_flags,
        "visible_marker_ids": visible_marker_ids,
        "solved_marker_ids": solved_marker_ids,
        "used_marker_ids": sorted(used_marker_set),
        "unused_solved_marker_ids": unused_solved_ids,
        "world_from_camera": world_pose,
        "previous_pose_frame": previous_pose_frame,
        "frames_since_previous_pose": frames_since_previous_pose,
        "delta_from_previous_world_pose": previous_delta,
        "candidate_consistency": {
            "translation_spread_mm": None if trans_mm is None else float(trans_mm),
            "rotation_spread_deg": None if rot_deg is None else float(rot_deg),
        },
        "robust_filter": robust_diagnostics,
        "mean_reprojection_error_px": (
            None
            if not detected_poses
            else float(np.mean([pose["reprojection_error"] for pose in detected_poses.values()]))
        ),
        "per_marker": per_marker,
    }


def draw_graph_panel(canvas, history, rect, title, series, value_unit):
    x0, y0, width, height = rect
    cv2.rectangle(canvas, (x0, y0), (x0 + width, y0 + height), (36, 36, 36), -1)
    cv2.rectangle(canvas, (x0, y0), (x0 + width, y0 + height), (80, 80, 80), 1)

    plot_left = x0 + 70
    plot_right = x0 + width - 18
    plot_top = y0 + 34
    plot_bottom = y0 + height - 34
    plot_width = max(1, plot_right - plot_left)
    plot_height = max(1, plot_bottom - plot_top)

    for index in range(5):
        gy = int(plot_top + index * plot_height / 4.0)
        cv2.line(canvas, (plot_left, gy), (plot_right, gy), GRAPH_GRID_COLOR, 1)
    for index in range(6):
        gx = int(plot_left + index * plot_width / 5.0)
        cv2.line(canvas, (gx, plot_top), (gx, plot_bottom), GRAPH_GRID_COLOR, 1)

    cv2.putText(
        canvas,
        title,
        (x0 + 12, y0 + 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        GRAPH_TEXT_COLOR,
        1,
        cv2.LINE_AA,
    )

    values = []
    for item in history:
        for key, _, _ in series:
            value = item.get(key, np.nan)
            if np.isfinite(value):
                values.append(float(value))

    if values:
        low = float(np.min(values))
        high = float(np.max(values))
        if abs(high - low) < 1e-6:
            margin = max(1.0, abs(high) * 0.05)
            low -= margin
            high += margin
        else:
            margin = (high - low) * 0.08
            low -= margin
            high += margin
    else:
        low, high = -1.0, 1.0

    cv2.putText(
        canvas,
        f"{high:7.1f}{value_unit}",
        (x0 + 6, plot_top + 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        GRAPH_TEXT_COLOR,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"{low:7.1f}{value_unit}",
        (x0 + 6, plot_bottom),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        GRAPH_TEXT_COLOR,
        1,
        cv2.LINE_AA,
    )

    count = len(history)
    denom = max(1, GRAPH_HISTORY_FRAMES - 1)
    for key, label, color in series:
        previous = None
        for index, item in enumerate(history):
            value = item.get(key, np.nan)
            if not np.isfinite(value):
                previous = None
                continue
            px = int(plot_right - (count - 1 - index) * plot_width / denom)
            normalized = (float(value) - low) / (high - low)
            py = int(plot_bottom - np.clip(normalized, 0.0, 1.0) * plot_height)
            point = (px, py)
            if previous is not None:
                cv2.line(canvas, previous, point, color, 2, cv2.LINE_AA)
            previous = point

        latest = history[-1].get(key, np.nan) if history else np.nan
        text = f"{label}:{latest:7.1f}" if np.isfinite(latest) else f"{label}:   ---"
        legend_x = x0 + 235 + series.index((key, label, color)) * 160
        cv2.putText(
            canvas,
            text,
            (legend_x, y0 + 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            1,
            cv2.LINE_AA,
        )


def draw_verification_graph(history):
    canvas = np.full((GRAPH_HEIGHT, GRAPH_WIDTH, 3), GRAPH_BG_COLOR, dtype=np.uint8)
    if not history:
        return canvas

    panels = [
        (
            (18, 18, GRAPH_WIDTH - 36, 205),
            "World position, mm",
            [
                ("x", "X", (80, 220, 255)),
                ("y", "Y", (80, 255, 120)),
                ("z", "Z", (255, 180, 80)),
            ],
            "mm",
        ),
        (
            (18, 242, GRAPH_WIDTH - 36, 205),
            "World rotation, deg",
            [
                ("roll", "Roll", (255, 120, 120)),
                ("pitch", "Pitch", (190, 140, 255)),
                ("yaw", "Yaw", (120, 210, 255)),
            ],
            "deg",
        ),
        (
            (18, 466, GRAPH_WIDTH - 36, 205),
            "Consistency / artifacts",
            [
                ("trans", "Spread mm", (80, 220, 255)),
                ("rot", "Spread deg", (255, 180, 80)),
                ("err", "Err px", (255, 120, 120)),
                ("markers", "Markers", (160, 255, 160)),
            ],
            "",
        ),
    ]

    for rect, title, series, unit in panels:
        draw_graph_panel(canvas, history, rect, title, series, unit)

    cv2.putText(
        canvas,
        f"Last {len(history)} frames. Gaps mean pose was lost.",
        (18, GRAPH_HEIGHT - 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        GRAPH_TEXT_COLOR,
        1,
        cv2.LINE_AA,
    )
    return canvas


def run_learning_mode(cap, dictionary, camera_matrix, dist_coeffs, planar=False):
    marker_world_estimates = {
        BASE_MARKER_ID: {
            "transform": np.eye(4, dtype=np.float64),
            "count": 1,
        }
    }
    marker_error_ema = {}
    quality_trans_ema = None
    quality_rot_ema = None
    last_status = "Show marker 0 together with other markers."
    mode_name = "learning (planar)" if planar else "learning (3D)"
    current_fps = 0.0
    fps_frame_count = 0
    fps_started_at = time.perf_counter()
    previous_world_from_camera = None
    previous_pose_frame = None
    frame_index = 0
    filter_status = "Filter: waiting for measurements"

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

        marker_corners, marker_ids, detected_poses = detect_marker_poses(
            frame,
            dictionary,
            camera_matrix,
            dist_coeffs,
        )

        # update per-marker reprojection error EMA
        for mid, pose in detected_poses.items():
            update_error_ema(marker_error_ema, mid, pose["reprojection_error"])

        marker_counts = {mid: data["count"] for mid, data in marker_world_estimates.items()}
        draw_marker_visuals(
            frame, marker_corners, marker_ids, detected_poses,
            camera_matrix, dist_coeffs, marker_counts, marker_error_ema,
        )

        frames_since_previous_pose = (
            int(frame_index - previous_pose_frame) if previous_pose_frame is not None else None
        )
        world_from_camera, new_marker_ids, (trans_mm, rot_deg), diagnostics = update_marker_world_estimates(
            marker_world_estimates,
            detected_poses,
            planar=planar,
            previous_world_from_camera=previous_world_from_camera,
            marker_error_ema=marker_error_ema,
            frames_since_previous_pose=frames_since_previous_pose,
        )

        if trans_mm is not None:
            if quality_trans_ema is None:
                quality_trans_ema = trans_mm
                quality_rot_ema = rot_deg
            else:
                quality_trans_ema = EMA_ALPHA * trans_mm + (1.0 - EMA_ALPHA) * quality_trans_ema
                quality_rot_ema = EMA_ALPHA * rot_deg + (1.0 - EMA_ALPHA) * quality_rot_ema

        if new_marker_ids:
            last_status = f"Learned markers: {', '.join(str(v) for v in sorted(new_marker_ids))}"
        elif world_from_camera is None:
            if diagnostics["reason"] == "single_marker_jump":
                last_status = "Rejected single-marker jump; keep 2+ known markers in view."
            elif diagnostics["reason"] == "no_quality_candidates":
                last_status = f"Need marker {BASE_MARKER_ID} or another good learned marker in view."
            else:
                last_status = f"Need marker {BASE_MARKER_ID} or another already learned marker in view."

        rejected_reproj = (
            len(diagnostics["soft_rejected_by_reprojection_error"])
            + len(diagnostics["hard_rejected_by_reprojection_error"])
        )
        rejected_residual = len(diagnostics["rejected_by_residual"])
        rejected_updates = len(diagnostics["skipped_marker_updates"])
        filter_status = (
            f"Filter: accepted={diagnostics['accepted_marker_ids']} "
            f"rej_err={rejected_reproj} rej_res={rejected_residual} rej_upd={rejected_updates}"
        )

        if world_from_camera is not None:
            previous_world_from_camera = np.array(world_from_camera, dtype=np.float64, copy=True)
            previous_pose_frame = frame_index
            draw_world_pose_text(
                frame,
                "Camera pose from learned markers",
                world_from_camera,
                10,
                65,
                INFO_COLOR,
            )

        draw_quality_score(frame, quality_trans_ema, quality_rot_ema, 10, 360)

        draw_multiline_text(
            frame,
            [
                f"Mode: {mode_name}",
                f"Known markers: {', '.join(str(v) for v in sorted(marker_world_estimates))}",
                f"Status: {last_status}",
                filter_status,
                "Planar floor constraints: ON" if planar else "Planar floor constraints: OFF",
                "S - save marker_layout.json",
                "ESC - exit",
            ],
            10,
            390,
            INFO_COLOR,
        )
        cv2.putText(
            frame,
            f"FPS: {current_fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            TEXT_COLOR,
            2,
            cv2.LINE_AA,
        )

        frame_index += 1
        cv2.imshow(WINDOW_TITLE, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            save_marker_layout(marker_world_estimates, planar=planar)
            last_status = f"Saved {len(marker_world_estimates)} markers to {MARKER_LAYOUT_JSON_PATH.name}"
            print(last_status)
        if key == 27:
            break


def run_verification_mode(cap, dictionary, camera_matrix, dist_coeffs):
    layout_payload, marker_world_transforms, marker_counts = load_marker_layout()
    marker_length_mm = float(layout_payload.get("marker_length_mm", ARUCO_MARKER_LENGTH_MM))
    marker_error_ema = {}
    graph_history = deque(maxlen=GRAPH_HISTORY_FRAMES)
    quality_trans_ema = None
    quality_rot_ema = None
    current_fps = 0.0
    fps_frame_count = 0
    fps_started_at = time.perf_counter()
    last_status = f"Loaded {len(marker_world_transforms)} markers from {MARKER_LAYOUT_JSON_PATH.name}"
    layout_mode = layout_payload.get("layout_mode", "3d")
    log_path, log_file = create_verification_log_file()
    frame_index = 0
    previous_world_from_camera = None
    previous_pose_frame = None
    print(last_status)
    log_file.write(
        json.dumps(
            {
                "record_type": "session",
                "wall_time_unix": float(time.time()),
                "calibration_path": str(CALIBRATION_JSON_PATH),
                "marker_layout_path": str(MARKER_LAYOUT_JSON_PATH),
                "layout_mode": layout_mode,
                "map_marker_ids": sorted(int(value) for value in marker_world_transforms),
                "marker_length_mm": marker_length_mm,
                "dictionary_name": layout_payload.get("dictionary_name", DICTIONARY_NAME),
                "camera_source": CAMERA_SOURCE,
                "camera_backend": CAMERA_BACKEND,
                "target_width": TARGET_WIDTH,
                "target_height": TARGET_HEIGHT,
                "target_fps": TARGET_FPS,
                "artifact_jump_mm": ARTIFACT_JUMP_MM,
                "max_reprojection_error_px": MAX_REPROJECTION_ERROR_PX,
                "hard_max_reprojection_error_px": HARD_MAX_REPROJECTION_ERROR_PX,
                "robust_candidate_residual_mm": ROBUST_CANDIDATE_RESIDUAL_MM,
                "single_marker_max_jump_mm": SINGLE_MARKER_MAX_JUMP_MM,
                "dual_marker_max_spread_mm": DUAL_MARKER_MAX_SPREAD_MM,
                "ema_exclusion_threshold_px": EMA_EXCLUSION_THRESHOLD_PX,
                "long_pose_loss_frames": LONG_POSE_LOSS_FRAMES,
            },
            ensure_ascii=False,
        )
        + "\n"
    )

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

        marker_corners, marker_ids, detected_poses = detect_marker_poses(
            frame,
            dictionary,
            camera_matrix,
            dist_coeffs,
            marker_length_mm,
        )

        for mid, pose in detected_poses.items():
            update_error_ema(marker_error_ema, mid, pose["reprojection_error"])

        draw_marker_visuals(
            frame, marker_corners, marker_ids, detected_poses,
            camera_matrix, dist_coeffs, marker_counts, marker_error_ema,
        )

        frames_since_previous_pose = (
            int(frame_index - previous_pose_frame) if previous_pose_frame is not None else None
        )
        world_from_camera, used_marker_ids, (trans_mm, rot_deg), candidate_details, robust_diagnostics = estimate_world_from_camera_robust(
            marker_world_transforms,
            detected_poses,
            previous_world_from_camera,
            marker_error_ema=marker_error_ema,
            frames_since_previous_pose=frames_since_previous_pose,
        )

        if trans_mm is not None:
            if quality_trans_ema is None:
                quality_trans_ema = trans_mm
                quality_rot_ema = rot_deg
            else:
                quality_trans_ema = EMA_ALPHA * trans_mm + (1.0 - EMA_ALPHA) * quality_trans_ema
                quality_rot_ema = EMA_ALPHA * rot_deg + (1.0 - EMA_ALPHA) * quality_rot_ema

        if world_from_camera is not None:
            last_status = f"Using markers: {', '.join(str(v) for v in sorted(set(used_marker_ids)))}"
            draw_world_pose_text(
                frame,
                "Camera pose in marker map",
                world_from_camera,
                10,
                65,
                INFO_COLOR,
            )
        else:
            last_status = "No known map markers in view."

        graph_history.append(
            build_verification_graph_sample(
                world_from_camera,
                used_marker_ids,
                trans_mm,
                rot_deg,
                detected_poses,
            )
        )

        log_record = build_verification_log_record(
            frame_index,
            now,
            current_fps,
            layout_mode,
            marker_ids,
            detected_poses,
            marker_error_ema,
            world_from_camera,
            previous_world_from_camera,
            used_marker_ids,
            trans_mm,
            rot_deg,
            candidate_details,
            robust_diagnostics,
            previous_pose_frame,
        )
        log_file.write(json.dumps(log_record, ensure_ascii=False) + "\n")
        if frame_index % 30 == 0 or log_record["artifact_flags"]:
            log_file.flush()
        if world_from_camera is not None:
            previous_world_from_camera = np.array(world_from_camera, dtype=np.float64, copy=True)
            previous_pose_frame = frame_index
        frame_index += 1

        draw_quality_score(frame, quality_trans_ema, quality_rot_ema, 10, 360)

        draw_multiline_text(
            frame,
            [
                "Mode 3: verification",
                f"Map markers: {', '.join(str(v) for v in sorted(marker_world_transforms))}",
                f"Status: {last_status}",
                f"Base marker: {layout_payload['base_marker_id']}",
                f"Layout mode: {layout_mode}",
                f"Log: {log_path.name}",
                "ESC - exit",
            ],
            10,
            390,
            INFO_COLOR,
        )
        cv2.putText(
            frame,
            f"FPS: {current_fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            TEXT_COLOR,
            2,
            cv2.LINE_AA,
        )

        cv2.imshow(WINDOW_TITLE, frame)
        cv2.imshow(GRAPH_WINDOW_TITLE, draw_verification_graph(graph_history))
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

    log_file.close()
    print(f"Mode 3 log saved: {log_path.resolve()}")


def run_udp_monitor_mode():
    graph_history = deque(maxlen=GRAPH_HISTORY_FRAMES)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_LOG_LISTEN_IP, UDP_LOG_PORT))
    sock.settimeout(0.05)
    packet_count = 0
    bad_packet_count = 0
    last_packet = None
    last_packet_time = None
    print(f"Mode 4 UDP monitor: listening on {UDP_LOG_LISTEN_IP}:{UDP_LOG_PORT}")

    while True:
        try:
            data, addr = sock.recvfrom(65535)
        except socket.timeout:
            pass
        else:
            try:
                packet = json.loads(data.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                bad_packet_count += 1
            else:
                if packet.get("record_type") == "aruco_pose":
                    packet_count += 1
                    last_packet = packet
                    last_packet_time = time.time()
                    graph_history.append(build_udp_graph_sample(packet))
                else:
                    bad_packet_count += 1

        canvas = draw_verification_graph(graph_history)
        status_lines = [
            f"Mode 4 UDP monitor  {UDP_LOG_LISTEN_IP}:{UDP_LOG_PORT}",
            f"Packets: {packet_count}  bad: {bad_packet_count}",
        ]
        if last_packet is None:
            status_lines.append("Waiting for aruco_pose UDP packets...")
        else:
            age = time.time() - last_packet_time if last_packet_time is not None else 0.0
            status_lines.extend(
                [
                    f"Last age: {age:.2f}s  seq={last_packet.get('sequence')}",
                    f"Used: {last_packet.get('used_marker_ids', [])}",
                    f"Visible: {last_packet.get('visible_marker_ids', [])}",
                    (
                        f"Rejected: err={len(last_packet.get('rejected_by_reprojection_error', []))} "
                        f"res={len(last_packet.get('rejected_by_residual', []))} "
                        f"spread={len(last_packet.get('rejected_by_spread', []))}"
                    ),
                    "ESC - exit",
                ]
            )
        draw_multiline_text(canvas, status_lines, 24, 42, GRAPH_TEXT_COLOR)
        cv2.imshow(GRAPH_WINDOW_TITLE, canvas)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

    sock.close()


def main():
    mode = choose_mode()
    if mode is None:
        return

    if mode == UDP_MONITOR_MODE:
        run_udp_monitor_mode()
        return

    camera_matrix, dist_coeffs = load_calibration()
    dictionary = get_dictionary()
    cap = open_camera()
    if cap is None:
        print(f"Error: cannot open camera source {CAMERA_SOURCE}")
        return

    try:
        if mode == LEARNING_MODE_3D:
            run_learning_mode(cap, dictionary, camera_matrix, dist_coeffs, planar=False)
        elif mode == LEARNING_MODE_PLANAR:
            run_learning_mode(cap, dictionary, camera_matrix, dist_coeffs, planar=True)
        elif mode == VERIFICATION_MODE:
            if not MARKER_LAYOUT_JSON_PATH.exists():
                print(f"Error: {MARKER_LAYOUT_JSON_PATH} not found. Run a learning mode first.")
                return
            run_verification_mode(cap, dictionary, camera_matrix, dist_coeffs)
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
