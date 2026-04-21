import json
import sys
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
WINDOW_TITLE = "Group ArUco Detection"
SELECTION_WINDOW_TITLE = "Select Mode"

DICTIONARY_NAME = "DICT_APRILTAG_36h11"
ARUCO_MARKER_LENGTH_MM = 180.0
AXIS_LENGTH_MM = 60.0
BASE_MARKER_ID = 0

FPS_UPDATE_PERIOD_SEC = 0.5
TEXT_COLOR = (0, 255, 0)
INFO_COLOR = (255, 220, 0)
MARKER_COLOR = (0, 200, 255)
BASE_MARKER_COLOR = (0, 255, 0)
ERROR_COLOR = (80, 80, 255)
QUALITY_GOOD_COLOR = (0, 220, 0)
QUALITY_MED_COLOR = (0, 200, 220)
QUALITY_BAD_COLOR = (60, 60, 255)

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
    """Std-dev of camera positions across independent marker estimates — lower = better map."""
    if len(candidates) < 2:
        return None
    positions = np.array([t[:3, 3] for t in candidates], dtype=np.float64)
    return float(np.mean(np.std(positions, axis=0)))


def update_error_ema(ema_dict, marker_id, new_error):
    if marker_id in ema_dict:
        ema_dict[marker_id] = EMA_ALPHA * new_error + (1.0 - EMA_ALPHA) * ema_dict[marker_id]
    else:
        ema_dict[marker_id] = new_error


def parse_mode_arg():
    for arg in sys.argv[1:]:
        value = arg.strip().lower()
        if value in {"mode=1", "1", "--mode=1"}:
            return 1
        if value in {"mode=2", "2", "--mode=2"}:
            return 2
    return None


def select_mode_by_key():
    canvas = np.full((360, 900, 3), 30, dtype=np.uint8)
    lines = [
        "Select mode",
        "Press 1  - learning mode",
        "Press 2  - localization mode",
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
            return 1
        if key == ord("2"):
            cv2.destroyWindow(SELECTION_WINDOW_TITLE)
            return 2
        if key == 27:
            cv2.destroyWindow(SELECTION_WINDOW_TITLE)
            return None


def choose_mode():
    mode = parse_mode_arg()
    if mode is not None:
        return mode
    return select_mode_by_key()


def detect_marker_poses(frame, dictionary, camera_matrix, dist_coeffs):
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
            ARUCO_MARKER_LENGTH_MM,
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


def normalize_marker_world_estimates(marker_world_estimates):
    if BASE_MARKER_ID not in marker_world_estimates:
        raise RuntimeError(f"Base marker {BASE_MARKER_ID} is missing from learned map.")

    base_transform = marker_world_estimates[BASE_MARKER_ID]["transform"]
    normalization_transform = invert_transform(base_transform)

    normalized = {}
    for marker_id, marker_data in marker_world_estimates.items():
        normalized_transform = normalization_transform @ marker_data["transform"]
        normalized[marker_id] = {
            "transform": normalized_transform,
            "count": marker_data["count"],
        }

    normalized[BASE_MARKER_ID]["transform"] = np.eye(4, dtype=np.float64)
    return normalized


def save_marker_layout(marker_world_estimates):
    normalized_estimates = normalize_marker_world_estimates(marker_world_estimates)
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
        "markers": markers_payload,
    }
    MARKER_LAYOUT_JSON_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def update_marker_world_estimates(marker_world_estimates, detected_poses):
    world_from_camera_candidates = []
    for marker_id, pose in detected_poses.items():
        if marker_id not in marker_world_estimates:
            continue
        world_from_marker = marker_world_estimates[marker_id]["transform"]
        world_from_camera_candidates.append(
            world_from_marker @ invert_transform(pose["camera_from_marker"])
        )

    if not world_from_camera_candidates:
        return None, [], None

    consistency_mm = compute_pose_consistency_mm(world_from_camera_candidates)
    world_from_camera = average_transforms(world_from_camera_candidates)
    new_marker_ids = []
    for marker_id, pose in detected_poses.items():
        estimated_world_from_marker = world_from_camera @ pose["camera_from_marker"]
        if marker_id in marker_world_estimates:
            count = marker_world_estimates[marker_id]["count"]
            old_transform = marker_world_estimates[marker_id]["transform"]
            marker_world_estimates[marker_id]["transform"] = average_transforms(
                [old_transform] * count + [estimated_world_from_marker]
            )
            marker_world_estimates[marker_id]["count"] = count + 1
        else:
            marker_world_estimates[marker_id] = {
                "transform": estimated_world_from_marker,
                "count": 1,
            }
            new_marker_ids.append(marker_id)

    return world_from_camera, new_marker_ids, consistency_mm


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
    return average_transforms(world_from_camera_candidates), used_marker_ids, consistency_mm


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


def draw_quality_score(frame, quality_ema, x, y):
    if quality_ema is None:
        label = "Map quality: --- (need 2+ markers)"
        color = INFO_COLOR
    elif quality_ema < 5.0:
        label = f"Map quality: {quality_ema:.1f} mm  GOOD"
        color = QUALITY_GOOD_COLOR
    elif quality_ema < 20.0:
        label = f"Map quality: {quality_ema:.1f} mm  OK"
        color = QUALITY_MED_COLOR
    else:
        label = f"Map quality: {quality_ema:.1f} mm  POOR"
        color = QUALITY_BAD_COLOR
    cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)


def run_learning_mode(cap, dictionary, camera_matrix, dist_coeffs):
    marker_world_estimates = {
        BASE_MARKER_ID: {
            "transform": np.eye(4, dtype=np.float64),
            "count": 1,
        }
    }
    marker_error_ema = {}
    quality_ema = None
    last_status = "Show marker 0 together with other markers."
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

        world_from_camera, new_marker_ids, consistency_mm = update_marker_world_estimates(
            marker_world_estimates,
            detected_poses,
        )

        if consistency_mm is not None:
            if quality_ema is None:
                quality_ema = consistency_mm
            else:
                quality_ema = EMA_ALPHA * consistency_mm + (1.0 - EMA_ALPHA) * quality_ema

        if new_marker_ids:
            last_status = f"Learned markers: {', '.join(str(v) for v in sorted(new_marker_ids))}"
        elif world_from_camera is None:
            last_status = f"Need marker {BASE_MARKER_ID} or another already learned marker in view."

        if world_from_camera is not None:
            draw_world_pose_text(
                frame,
                "Camera pose from learned markers",
                world_from_camera,
                10,
                65,
                INFO_COLOR,
            )

        draw_quality_score(frame, quality_ema, 10, 360)

        draw_multiline_text(
            frame,
            [
                "Mode 1: learning",
                f"Known markers: {', '.join(str(v) for v in sorted(marker_world_estimates))}",
                f"Status: {last_status}",
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

        cv2.imshow(WINDOW_TITLE, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            save_marker_layout(marker_world_estimates)
            last_status = f"Saved {len(marker_world_estimates)} markers to {MARKER_LAYOUT_JSON_PATH.name}"
            print(last_status)
        if key == 27:
            break


def run_localization_mode(cap, dictionary, camera_matrix, dist_coeffs):
    layout_payload, marker_world_transforms, marker_counts = load_marker_layout()
    marker_error_ema = {}
    quality_ema = None
    current_fps = 0.0
    fps_frame_count = 0
    fps_started_at = time.perf_counter()
    last_status = f"Loaded {len(marker_world_transforms)} markers from {MARKER_LAYOUT_JSON_PATH.name}"
    print(last_status)

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

        for mid, pose in detected_poses.items():
            update_error_ema(marker_error_ema, mid, pose["reprojection_error"])

        draw_marker_visuals(
            frame, marker_corners, marker_ids, detected_poses,
            camera_matrix, dist_coeffs, marker_counts, marker_error_ema,
        )

        world_from_camera, used_marker_ids, consistency_mm = estimate_world_from_camera(
            marker_world_transforms,
            detected_poses,
        )

        if consistency_mm is not None:
            if quality_ema is None:
                quality_ema = consistency_mm
            else:
                quality_ema = EMA_ALPHA * consistency_mm + (1.0 - EMA_ALPHA) * quality_ema

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

        draw_quality_score(frame, quality_ema, 10, 360)

        draw_multiline_text(
            frame,
            [
                "Mode 2: localization",
                f"Map markers: {', '.join(str(v) for v in sorted(marker_world_transforms))}",
                f"Status: {last_status}",
                f"Base marker: {layout_payload['base_marker_id']}",
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
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break


def main():
    mode = choose_mode()
    if mode is None:
        return

    camera_matrix, dist_coeffs = load_calibration()
    dictionary = get_dictionary()
    cap = open_camera()
    if cap is None:
        print(f"Error: cannot open camera source {CAMERA_SOURCE}")
        return

    try:
        if mode == 1:
            run_learning_mode(cap, dictionary, camera_matrix, dist_coeffs)
        elif mode == 2:
            if not MARKER_LAYOUT_JSON_PATH.exists():
                print(f"Error: {MARKER_LAYOUT_JSON_PATH} not found. Run mode 1 first.")
                return
            run_localization_mode(cap, dictionary, camera_matrix, dist_coeffs)
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
