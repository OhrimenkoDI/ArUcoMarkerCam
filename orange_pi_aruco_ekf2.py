from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from pymavlink import mavutil


SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_CALIBRATION_PATH = SCRIPT_DIR / "camera_calibration.json"
DEFAULT_LAYOUT_PATH = SCRIPT_DIR / "marker_layout.json"

CAMERA_PROFILES: dict[str, dict[str, int | str]] = {
    "rys": {
        "source": "/dev/video1",
        "width": 1280,
        "height": 720,
        "fps": 30,
    },
    "usb": {
        "source": "/dev/video4",
        "width": 1280,
        "height": 720,
        "fps": 30,
    },
}

RC_OVERRIDE_INTERVAL_SEC = 0.02
ODOMETRY_SEND_HZ = 20.0
ODOMETRY_PERIOD_SEC = 1.0 / ODOMETRY_SEND_HZ
DIAG_PERIOD_SEC = 0.5
EMA_ALPHA = 0.1


def log(message: str) -> None:
    print(f"[ArucoEKF2] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Orange Pi 5 Ultra: ArUco localization -> ArduPilot EKF Source Set 2"
    )
    parser.add_argument("--calibration", default=str(DEFAULT_CALIBRATION_PATH))
    parser.add_argument("--layout", default=str(DEFAULT_LAYOUT_PATH))
    parser.add_argument(
        "--camera-profile",
        choices=sorted(CAMERA_PROFILES),
        default="rys",
        help="Orange Pi camera profile.",
    )
    parser.add_argument("--source", help="Override camera source, for example /dev/video1.")
    parser.add_argument("--width", type=int, help="Capture width.")
    parser.add_argument("--height", type=int, help="Capture height.")
    parser.add_argument("--fps", type=int, help="Capture FPS.")
    parser.add_argument("--dict-name", default="DICT_APRILTAG_36h11")
    parser.add_argument("--marker-length-mm", type=float, default=168.0)
    parser.add_argument("--mavlink", default="/dev/ttyS3,921600")
    parser.add_argument(
        "--ekf-rc-channel",
        type=int,
        default=7,
        help="RC channel with RCx_OPTION=90.",
    )
    parser.add_argument(
        "--ekf-source-pwm",
        type=int,
        default=1500,
        help="PWM that selects EKF Source Set 2.",
    )
    parser.add_argument(
        "--disable-rc-override",
        action="store_true",
        help="Do not send RC channel override for source selection.",
    )
    return parser.parse_args()


def load_calibration(calibration_path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = json.loads(calibration_path.read_text(encoding="utf-8"))
    camera_matrix = np.array(payload["camera_matrix"], dtype=np.float64)
    dist_coeffs = np.array(payload["dist_coeffs"], dtype=np.float64)
    return camera_matrix, dist_coeffs


def get_dictionary(dictionary_name: str):
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_name))


def build_marker_object_points(marker_length_mm: float) -> np.ndarray:
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


def solve_marker_pose(
    corners: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    marker_length_mm: float,
) -> tuple[bool, np.ndarray, np.ndarray, float]:
    image_points = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    object_points = build_marker_object_points(marker_length_mm)
    ok, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    if not ok:
        return False, np.zeros((3, 1)), np.zeros((3, 1)), float("inf")
    projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs)
    reprojection_error = float(
        np.mean(np.linalg.norm(image_points - projected.reshape(4, 2), axis=1))
    )
    return True, rvec, tvec, reprojection_error


def rt_to_transform(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    rotation_matrix, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation_matrix
    transform[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
    return transform


def invert_transform(transform: np.ndarray) -> np.ndarray:
    inverse = np.eye(4, dtype=np.float64)
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -(rotation.T @ translation)
    return inverse


def average_transforms(transforms: list[np.ndarray]) -> np.ndarray:
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


def compute_pose_consistency_mm(candidates: list[np.ndarray]) -> float | None:
    if len(candidates) < 2:
        return None
    positions = np.array([transform[:3, 3] for transform in candidates], dtype=np.float64)
    return float(np.mean(np.std(positions, axis=0)))


def load_marker_layout(layout_path: Path) -> tuple[dict, dict[int, np.ndarray], dict[int, int]]:
    payload = json.loads(layout_path.read_text(encoding="utf-8"))
    marker_transforms: dict[int, np.ndarray] = {}
    marker_counts: dict[int, int] = {}
    for marker_id_text, marker_payload in payload["markers"].items():
        marker_id = int(marker_id_text)
        rvec = np.array(marker_payload["rvec"], dtype=np.float64).reshape(3, 1)
        tvec = np.array(marker_payload["tvec"], dtype=np.float64).reshape(3, 1)
        marker_transforms[marker_id] = rt_to_transform(rvec, tvec)
        marker_counts[marker_id] = int(marker_payload.get("observation_count", 0))
    return payload, marker_transforms, marker_counts


def detect_marker_poses(
    frame: np.ndarray,
    dictionary,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    marker_length_mm: float,
) -> dict[int, dict[str, np.ndarray | float]]:
    detector = cv2.aruco.ArucoDetector(dictionary)
    marker_corners, marker_ids, _ = detector.detectMarkers(frame)
    poses: dict[int, dict[str, np.ndarray | float]] = {}
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
                "camera_from_marker": rt_to_transform(rvec, tvec),
                "reprojection_error_px": reprojection_error,
            }
    return poses


def estimate_world_from_camera(
    marker_world_transforms: dict[int, np.ndarray],
    detected_poses: dict[int, dict[str, np.ndarray | float]],
) -> tuple[np.ndarray | None, list[int], float | None]:
    candidates: list[np.ndarray] = []
    used_marker_ids: list[int] = []
    for marker_id, pose in detected_poses.items():
        if marker_id not in marker_world_transforms:
            continue
        candidates.append(
            marker_world_transforms[marker_id] @ invert_transform(pose["camera_from_marker"])
        )
        used_marker_ids.append(marker_id)

    if not candidates:
        return None, [], None

    return average_transforms(candidates), used_marker_ids, compute_pose_consistency_mm(candidates)


_CV_TO_FRD = np.array(
    [
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)


def rotation_matrix_to_quaternion_wxyz(rotation_matrix: np.ndarray) -> np.ndarray:
    trace = float(np.trace(rotation_matrix))
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) * s
        y = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) * s
        z = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) * s
    else:
        if rotation_matrix[0, 0] > rotation_matrix[1, 1] and rotation_matrix[0, 0] > rotation_matrix[2, 2]:
            s = 2.0 * np.sqrt(1.0 + rotation_matrix[0, 0] - rotation_matrix[1, 1] - rotation_matrix[2, 2])
            w = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / s
            x = 0.25 * s
            y = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / s
            z = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / s
        elif rotation_matrix[1, 1] > rotation_matrix[2, 2]:
            s = 2.0 * np.sqrt(1.0 + rotation_matrix[1, 1] - rotation_matrix[0, 0] - rotation_matrix[2, 2])
            w = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / s
            x = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / s
            y = 0.25 * s
            z = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + rotation_matrix[2, 2] - rotation_matrix[0, 0] - rotation_matrix[1, 1])
            w = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / s
            x = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / s
            y = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / s
            z = 0.25 * s

    quat = np.array([w, x, y, z], dtype=np.float64)
    norm = np.linalg.norm(quat)
    if norm <= 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return (quat / norm).astype(np.float32)


def rotation_matrix_to_euler_rpy(rotation_matrix: np.ndarray) -> np.ndarray:
    sy = float(np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2))
    singular = sy < 1e-6

    if not singular:
        roll = float(np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2]))
        pitch = float(np.arctan2(-rotation_matrix[2, 0], sy))
        yaw = float(np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0]))
    else:
        roll = float(np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1]))
        pitch = float(np.arctan2(-rotation_matrix[2, 0], sy))
        yaw = 0.0

    return np.array([roll, pitch, yaw], dtype=np.float32)


def world_from_camera_to_frd(world_from_camera: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    position_cv_mm = world_from_camera[:3, 3]
    position_frd_m = (_CV_TO_FRD @ position_cv_mm) / 1000.0

    rotation_cv = world_from_camera[:3, :3]
    rotation_frd = _CV_TO_FRD @ rotation_cv @ _CV_TO_FRD.T
    quaternion_wxyz = rotation_matrix_to_quaternion_wxyz(rotation_frd)
    return position_frd_m.astype(np.float32), quaternion_wxyz


def open_capture(source: str, width: int, height: int, fps: int) -> cv2.VideoCapture:
    pipeline = (
        f"v4l2src device={source} io-mode=2 ! "
        f"image/jpeg,width={width},height={height},framerate={fps}/1 ! "
        "jpegdec ! videoconvert ! appsink drop=true max-buffers=1 sync=false"
    )
    attempts = [
        (pipeline, cv2.CAP_GSTREAMER),
        (source, cv2.CAP_V4L2),
        (source, cv2.CAP_ANY),
    ]
    for candidate, backend in attempts:
        cap = cv2.VideoCapture(candidate, backend)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, fps)
            for _ in range(5):
                cap.read()
            log(
                f"Capture opened via backend={backend} source={candidate!r} "
                f"requested={width}x{height}@{fps}"
            )
            return cap
    raise RuntimeError(f"Could not open camera source {source}")


def connect_mavlink(connection_string: str):
    conn = mavutil.mavlink_connection(connection_string, force_connected=True)
    log(f"MAVLink connected: {connection_string}")
    return conn


def send_rc_override(conn, channel: int, pwm: int) -> None:
    if not 1 <= channel <= 8:
        raise ValueError(f"RC override supports channels 1..8 in this pymavlink build, got {channel}")
    values = [0] * 8
    values[channel - 1] = pwm
    conn.mav.rc_channels_override_send(
        1,
        1,
        *values,
    )


def send_odometry(conn, timestamp_us: int, position_frd_m: np.ndarray, quaternion_wxyz: np.ndarray, quality: int) -> None:
    nan_value = float("nan")
    if hasattr(conn.mav, "odometry_send"):
        conn.mav.odometry_send(
            time_usec=timestamp_us,
            frame_id=mavutil.mavlink.MAV_FRAME_LOCAL_FRD,
            child_frame_id=mavutil.mavlink.MAV_FRAME_BODY_FRD,
            x=float(position_frd_m[0]),
            y=float(position_frd_m[1]),
            z=float(position_frd_m[2]),
            q=[
                float(quaternion_wxyz[0]),
                float(quaternion_wxyz[1]),
                float(quaternion_wxyz[2]),
                float(quaternion_wxyz[3]),
            ],
            vx=nan_value,
            vy=nan_value,
            vz=nan_value,
            rollspeed=nan_value,
            pitchspeed=nan_value,
            yawspeed=nan_value,
            pose_covariance=[0.0] * 21,
            velocity_covariance=[0.0] * 21,
            reset_counter=0,
            estimator_type=mavutil.mavlink.MAV_ESTIMATOR_TYPE_VISION,
            quality=int(quality),
        )
        return

    rotation_matrix = np.array(
        [
            [
                1.0 - 2.0 * (quaternion_wxyz[2] ** 2 + quaternion_wxyz[3] ** 2),
                2.0 * (quaternion_wxyz[1] * quaternion_wxyz[2] - quaternion_wxyz[3] * quaternion_wxyz[0]),
                2.0 * (quaternion_wxyz[1] * quaternion_wxyz[3] + quaternion_wxyz[2] * quaternion_wxyz[0]),
            ],
            [
                2.0 * (quaternion_wxyz[1] * quaternion_wxyz[2] + quaternion_wxyz[3] * quaternion_wxyz[0]),
                1.0 - 2.0 * (quaternion_wxyz[1] ** 2 + quaternion_wxyz[3] ** 2),
                2.0 * (quaternion_wxyz[2] * quaternion_wxyz[3] - quaternion_wxyz[1] * quaternion_wxyz[0]),
            ],
            [
                2.0 * (quaternion_wxyz[1] * quaternion_wxyz[3] - quaternion_wxyz[2] * quaternion_wxyz[0]),
                2.0 * (quaternion_wxyz[2] * quaternion_wxyz[3] + quaternion_wxyz[1] * quaternion_wxyz[0]),
                1.0 - 2.0 * (quaternion_wxyz[1] ** 2 + quaternion_wxyz[2] ** 2),
            ],
        ],
        dtype=np.float64,
    )
    roll, pitch, yaw = rotation_matrix_to_euler_rpy(rotation_matrix)
    conn.mav.vision_position_estimate_send(
        timestamp_us,
        float(position_frd_m[0]),
        float(position_frd_m[1]),
        float(position_frd_m[2]),
        float(roll),
        float(pitch),
        float(yaw),
    )


def validate_paths(calibration_path: Path, layout_path: Path) -> None:
    if not calibration_path.exists():
        raise FileNotFoundError(
            f"Calibration file not found: {calibration_path}. "
            "Copy camera_calibration.json next to the script or pass --calibration."
        )
    if not layout_path.exists():
        raise FileNotFoundError(
            f"Marker layout file not found: {layout_path}. "
            "Create it in detect_charuco_aruco_group.py mode 1 first."
        )


def main() -> int:
    args = parse_args()
    calibration_path = Path(args.calibration).resolve()
    layout_path = Path(args.layout).resolve()
    validate_paths(calibration_path, layout_path)

    profile = dict(CAMERA_PROFILES[args.camera_profile])
    source = str(args.source or profile["source"])
    width = int(args.width or profile["width"])
    height = int(args.height or profile["height"])
    fps = int(args.fps or profile["fps"])

    camera_matrix, dist_coeffs = load_calibration(calibration_path)
    dictionary = get_dictionary(args.dict_name)
    layout_payload, marker_world_transforms, _ = load_marker_layout(layout_path)

    log(
        f"Loaded map with {len(marker_world_transforms)} markers, "
        f"base_marker_id={layout_payload.get('base_marker_id')}"
    )

    cap = open_capture(source, width, height, fps)
    conn = connect_mavlink(args.mavlink)

    stop_requested = False

    def handle_stop(_signum, _frame) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, handle_stop)
    signal.signal(signal.SIGTERM, handle_stop)

    marker_error_ema: dict[int, float] = {}
    quality_ema: float | None = None
    last_rc_override_time = 0.0
    last_odometry_time = 0.0
    last_diag_time = 0.0
    started_at = time.perf_counter()
    frame_count = 0
    odom_count = 0

    try:
        while not stop_requested:
            ok, frame = cap.read()
            if not ok:
                log("Frame read error")
                break

            now = time.perf_counter()
            frame_count += 1
            detected_poses = detect_marker_poses(
                frame,
                dictionary,
                camera_matrix,
                dist_coeffs,
                args.marker_length_mm,
            )

            for marker_id, pose in detected_poses.items():
                error_px = float(pose["reprojection_error_px"])
                previous = marker_error_ema.get(marker_id, error_px)
                marker_error_ema[marker_id] = EMA_ALPHA * error_px + (1.0 - EMA_ALPHA) * previous

            world_from_camera, used_marker_ids, consistency_mm = estimate_world_from_camera(
                marker_world_transforms,
                detected_poses,
            )

            if consistency_mm is not None:
                if quality_ema is None:
                    quality_ema = consistency_mm
                else:
                    quality_ema = EMA_ALPHA * consistency_mm + (1.0 - EMA_ALPHA) * quality_ema

            if not args.disable_rc_override and (now - last_rc_override_time) >= RC_OVERRIDE_INTERVAL_SEC:
                send_rc_override(conn, args.ekf_rc_channel, args.ekf_source_pwm)
                last_rc_override_time = now

            if world_from_camera is not None and (now - last_odometry_time) >= ODOMETRY_PERIOD_SEC:
                position_frd_m, quaternion_wxyz = world_from_camera_to_frd(world_from_camera)
                quality = min(100, max(1, len(used_marker_ids) * 25))
                send_odometry(conn, int(time.time() * 1e6), position_frd_m, quaternion_wxyz, quality)
                last_odometry_time = now
                odom_count += 1

            if (now - last_diag_time) >= DIAG_PERIOD_SEC:
                elapsed = max(now - started_at, 1e-6)
                capture_fps = frame_count / elapsed
                odom_hz = odom_count / elapsed
                if world_from_camera is None:
                    log(
                        f"capture={capture_fps:.1f}fps odom={odom_hz:.1f}Hz "
                        f"pose=NO visible={sorted(detected_poses)} map={sorted(marker_world_transforms)}"
                    )
                else:
                    position_frd_m, quaternion_wxyz = world_from_camera_to_frd(world_from_camera)
                    error_text = " ".join(
                        f"{marker_id}:{marker_error_ema[marker_id]:.1f}px"
                        for marker_id in sorted(used_marker_ids)
                        if marker_id in marker_error_ema
                    )
                    quality_text = "---" if quality_ema is None else f"{quality_ema:.1f}mm"
                    log(
                        f"capture={capture_fps:.1f}fps odom={odom_hz:.1f}Hz "
                        f"pos_frd=({position_frd_m[0]:+.3f},{position_frd_m[1]:+.3f},{position_frd_m[2]:+.3f})m "
                        f"q=({quaternion_wxyz[0]:+.3f},{quaternion_wxyz[1]:+.3f},{quaternion_wxyz[2]:+.3f},{quaternion_wxyz[3]:+.3f}) "
                        f"markers={sorted(set(used_marker_ids))} quality={quality_text} err=[{error_text}]"
                    )
                last_diag_time = now
    finally:
        if not args.disable_rc_override:
            for _ in range(5):
                send_rc_override(conn, args.ekf_rc_channel, 0)
                time.sleep(RC_OVERRIDE_INTERVAL_SEC)
        cap.release()
        conn.close()
        log("Stopped")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[ArucoEKF2] Error: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1)
