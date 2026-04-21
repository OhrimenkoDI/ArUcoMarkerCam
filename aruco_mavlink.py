"""
ArUco indoor localization → MAVLink ODOMETRY sender.
Headless (no GUI), designed for Orange Pi 5 Ultra.

World frame (OpenCV, marker-0 origin):
    X — right, Y — down, Z — forward  (right-handed)

MAVLink ODOMETRY frame_id = MAV_FRAME_LOCAL_FRD (20):
    X — forward, Y — right, Z — down

Conversion:  frd_x = cv_z,  frd_y = cv_x,  frd_z = cv_y

ArduPilot EKF3 parameters to set:
    AHRS_EKF_TYPE  = 3      (use EKF3)
    EK2_ENABLE     = 0
    EK3_ENABLE     = 1
    EK3_SRC1_POSXY = 6      (ExternalNav)
    EK3_SRC1_POSZ  = 6      (ExternalNav)
    EK3_SRC1_VELXY = 0      (None — we do not send velocity)
    EK3_SRC1_VELZ  = 0
    EK3_SRC1_YAW   = 6      (ExternalNav)
    EK3_GPS_CHECK  = 0      (skip GPS checks for indoor)
    GPS_TYPE       = 0      (disable GPS)
    COMPASS_USE    = 0      (optional, disable compass indoor)
    VISO_TYPE      = 0      (we use raw ODOMETRY, not T265 driver)

Dependencies:
    pip install pymavlink scipy opencv-contrib-python numpy
"""

import json
import signal
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from pymavlink import mavutil

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CALIBRATION_JSON_PATH = Path("camera_calibration.json")
MARKER_LAYOUT_JSON_PATH = Path("marker_layout.json")

# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------
CAMERA_SOURCE = "/dev/v4l/by-id/usb-RYS_HFR_USB2.0_Camera_RYS_HFR_USB2.0_Camera-video-index0"
CAMERA_BACKEND = "v4l2"   # use "auto" on Windows; "v4l2" on Linux/OrangePi
TARGET_WIDTH = 1280
TARGET_HEIGHT = 720
TARGET_FPS = 30
TARGET_FOURCC = "MJPG"

# ---------------------------------------------------------------------------
# ArUco
# ---------------------------------------------------------------------------
DICTIONARY_NAME = "DICT_4X4_50"
ARUCO_MARKER_LENGTH_MM = 158.0

# ---------------------------------------------------------------------------
# MAVLink
# ---------------------------------------------------------------------------
MAVLINK_CONNECTION = "udpout:127.0.0.1:14550"   # SITL
# For real flight controller via UART on Orange Pi:
# MAVLINK_CONNECTION = "/dev/ttyS3"  (check your board pinout)
# MAVLINK_BAUD = 921600

ODOMETRY_SEND_HZ = 20          # target send frequency
ODOMETRY_PERIOD_SEC = 1.0 / ODOMETRY_SEND_HZ

# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
DIAG_PRINT_HZ = 2              # terminal print frequency
EMA_ALPHA = 0.1                # smoothing for reprojection error & quality

BACKEND_MAP = {
    "auto": cv2.CAP_ANY,
    "dshow": cv2.CAP_DSHOW,
    "v4l2": cv2.CAP_V4L2,
}


# ===========================================================================
# Camera
# ===========================================================================

def load_calibration():
    payload = json.loads(CALIBRATION_JSON_PATH.read_text(encoding="utf-8"))
    camera_matrix = np.array(payload["camera_matrix"], dtype=np.float64)
    dist_coeffs = np.array(payload["dist_coeffs"], dtype=np.float64)
    return camera_matrix, dist_coeffs


def get_dictionary():
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, DICTIONARY_NAME))


def open_camera():
    params = [
        cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*TARGET_FOURCC),
        cv2.CAP_PROP_FRAME_WIDTH, TARGET_WIDTH,
        cv2.CAP_PROP_FRAME_HEIGHT, TARGET_HEIGHT,
        cv2.CAP_PROP_FPS, TARGET_FPS,
        cv2.CAP_PROP_BUFFERSIZE, 1,
    ]
    backend = BACKEND_MAP.get(CAMERA_BACKEND, cv2.CAP_ANY)
    cap = cv2.VideoCapture(CAMERA_SOURCE, backend, params)
    if not cap.isOpened():
        return None
    for _ in range(10):
        cap.read()
    return cap


# ===========================================================================
# ArUco detection & pose
# ===========================================================================

def build_marker_object_points(marker_length_mm):
    half = marker_length_mm * 0.5
    return np.array([
        [-half,  half, 0.0],
        [ half,  half, 0.0],
        [ half, -half, 0.0],
        [-half, -half, 0.0],
    ], dtype=np.float32)


def solve_marker_pose(corners, camera_matrix, dist_coeffs):
    image_pts = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    obj_pts = build_marker_object_points(ARUCO_MARKER_LENGTH_MM)
    ok, rvec, tvec = cv2.solvePnP(
        obj_pts, image_pts, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    if ok:
        projected, _ = cv2.projectPoints(obj_pts, rvec, tvec, camera_matrix, dist_coeffs)
        err = float(np.mean(np.linalg.norm(image_pts - projected.reshape(4, 2), axis=1)))
    else:
        err = float("inf")
    return ok, rvec, tvec, err


def rt_to_transform(rvec, tvec):
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
    return T


def invert_transform(T):
    R, t = T[:3, :3], T[:3, 3]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -(R.T @ t)
    return Ti


def detect_marker_poses(frame, dictionary, camera_matrix, dist_coeffs):
    detector = cv2.aruco.ArucoDetector(dictionary)
    corners_list, ids, _ = detector.detectMarkers(frame)
    poses = {}
    if ids is None or len(ids) == 0:
        return poses
    for corners, mid in zip(corners_list, ids.flatten()):
        ok, rvec, tvec, err = solve_marker_pose(corners, camera_matrix, dist_coeffs)
        if ok:
            poses[int(mid)] = {
                "camera_from_marker": rt_to_transform(rvec, tvec),
                "reprojection_error_px": err,
            }
    return poses


# ===========================================================================
# Map & localization
# ===========================================================================

def load_marker_layout():
    payload = json.loads(MARKER_LAYOUT_JSON_PATH.read_text(encoding="utf-8"))
    marker_transforms = {}
    marker_counts = {}
    for mid_str, data in payload["markers"].items():
        mid = int(mid_str)
        rvec = np.array(data["rvec"], dtype=np.float64).reshape(3, 1)
        tvec = np.array(data["tvec"], dtype=np.float64).reshape(3, 1)
        marker_transforms[mid] = rt_to_transform(rvec, tvec)
        marker_counts[mid] = int(data.get("observation_count", 0))
    return payload, marker_transforms, marker_counts


def average_transforms(transforms):
    translations = np.array([t[:3, 3] for t in transforms], dtype=np.float64)
    rotvecs = [cv2.Rodrigues(t[:3, :3])[0].reshape(3) for t in transforms]
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = np.mean(translations, axis=0)
    T[:3, :3], _ = cv2.Rodrigues(
        np.mean(np.array(rotvecs, dtype=np.float64), axis=0).reshape(3, 1)
    )
    return T


def estimate_world_from_camera(marker_world_transforms, detected_poses):
    candidates = []
    used_ids = []
    for mid, pose in detected_poses.items():
        if mid not in marker_world_transforms:
            continue
        candidates.append(
            marker_world_transforms[mid] @ invert_transform(pose["camera_from_marker"])
        )
        used_ids.append(mid)
    if not candidates:
        return None, [], None
    consistency_mm = None
    if len(candidates) >= 2:
        positions = np.array([c[:3, 3] for c in candidates])
        consistency_mm = float(np.mean(np.std(positions, axis=0)))
    return average_transforms(candidates), used_ids, consistency_mm


# ===========================================================================
# Coordinate conversion: OpenCV world → MAVLink FRD
# ===========================================================================

# OpenCV world frame: X right, Y down, Z forward
# MAVLink FRD local:  X forward, Y right, Z down
# Permutation: frd_x = cv_z, frd_y = cv_x, frd_z = cv_y
_CV_TO_FRD = np.array([
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
], dtype=np.float64)


def world_from_camera_to_frd(world_from_camera):
    """
    Returns:
        pos_frd_m  : np.array [3] position in meters, FRD frame
        q_wxyz     : np.array [4] quaternion [w, x, y, z], local FRD → body FRD
    """
    T = _CV_TO_FRD
    pos_cv_mm = world_from_camera[:3, 3]
    pos_frd_m = (T @ pos_cv_mm) / 1000.0

    R_cv = world_from_camera[:3, :3]
    R_frd = T @ R_cv @ T.T

    # scipy returns [x, y, z, w]; MAVLink wants [w, x, y, z]
    q_xyzw = Rotation.from_matrix(R_frd).as_quat()
    q_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float32)

    return pos_frd_m.astype(np.float32), q_wxyz


# ===========================================================================
# MAVLink
# ===========================================================================

def connect_mavlink():
    conn = mavutil.mavlink_connection(MAVLINK_CONNECTION)
    print(f"[MAVLink] connected: {MAVLINK_CONNECTION}")
    return conn


def send_odometry(conn, timestamp_us, pos_frd_m, q_wxyz, n_markers):
    NAN = float("nan")
    # quality: 0–100, scale by marker count (cap at 4 markers = 100)
    quality = min(100, n_markers * 25)
    conn.mav.odometry_send(
        time_usec=int(timestamp_us),
        frame_id=mavutil.mavlink.MAV_FRAME_LOCAL_FRD,       # position frame
        child_frame_id=mavutil.mavlink.MAV_FRAME_BODY_FRD,  # velocity/attitude frame
        x=float(pos_frd_m[0]),
        y=float(pos_frd_m[1]),
        z=float(pos_frd_m[2]),
        q=[float(q_wxyz[0]), float(q_wxyz[1]), float(q_wxyz[2]), float(q_wxyz[3])],
        vx=NAN, vy=NAN, vz=NAN,
        rollspeed=NAN, pitchspeed=NAN, yawspeed=NAN,
        pose_covariance=[0.0] * 21,
        velocity_covariance=[0.0] * 21,
        reset_counter=0,
        estimator_type=mavutil.mavlink.MAV_ESTIMATOR_TYPE_VISION,
        quality=quality,
    )


# ===========================================================================
# EMA helper
# ===========================================================================

def ema_update(ema_dict, key, value):
    if key in ema_dict:
        ema_dict[key] = EMA_ALPHA * value + (1.0 - EMA_ALPHA) * ema_dict[key]
    else:
        ema_dict[key] = value
    return ema_dict[key]


# ===========================================================================
# Main loop
# ===========================================================================

def run(cap, dictionary, camera_matrix, dist_coeffs, conn, marker_world_transforms, marker_counts):
    marker_error_ema = {}
    quality_ema = None
    last_send_t = 0.0
    last_print_t = 0.0
    print_period = 1.0 / DIAG_PRINT_HZ

    frame_count = 0
    send_count = 0
    loop_t = time.perf_counter()

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[Camera] frame read error", flush=True)
            break

        now = time.perf_counter()
        frame_count += 1

        detected_poses = detect_marker_poses(frame, dictionary, camera_matrix, dist_coeffs)

        for mid, pose in detected_poses.items():
            ema_update(marker_error_ema, mid, pose["reprojection_error_px"])

        world_from_camera, used_ids, consistency_mm = estimate_world_from_camera(
            marker_world_transforms, detected_poses
        )

        if consistency_mm is not None:
            quality_ema = ema_update({"q": quality_ema if quality_ema is not None else consistency_mm},
                                     "q", consistency_mm)

        # --- send at target frequency ---
        if world_from_camera is not None and (now - last_send_t) >= ODOMETRY_PERIOD_SEC:
            pos_frd_m, q_wxyz = world_from_camera_to_frd(world_from_camera)
            timestamp_us = int(time.time() * 1e6)
            send_odometry(conn, timestamp_us, pos_frd_m, q_wxyz, len(used_ids))
            last_send_t = now
            send_count += 1

        # --- terminal diagnostics ---
        if (now - last_print_t) >= print_period:
            elapsed = now - loop_t
            cam_fps = frame_count / elapsed if elapsed > 0 else 0.0
            mav_hz = send_count / elapsed if elapsed > 0 else 0.0

            if world_from_camera is not None:
                pos = world_from_camera[:3, 3] / 1000.0   # meters, OpenCV frame
                pos_frd_m, q_wxyz = world_from_camera_to_frd(world_from_camera)
                q_str = f"[{q_wxyz[0]:+.3f} {q_wxyz[1]:+.3f} {q_wxyz[2]:+.3f} {q_wxyz[3]:+.3f}]"
                err_parts = [f"{mid}:{marker_error_ema[mid]:.1f}px"
                             for mid in sorted(used_ids) if mid in marker_error_ema]
                qual_str = f"{quality_ema:.1f}mm" if quality_ema is not None else "---"
                print(
                    f"cam={cam_fps:4.1f}fps  mav={mav_hz:4.1f}Hz"
                    f"  XYZ[m] fwd={pos_frd_m[0]:+6.3f} rgt={pos_frd_m[1]:+6.3f} dwn={pos_frd_m[2]:+6.3f}"
                    f"  q={q_str}"
                    f"  err=[{' '.join(err_parts)}]"
                    f"  quality={qual_str}"
                    f"  markers={','.join(str(v) for v in sorted(used_ids))}",
                    flush=True,
                )
            else:
                visible = sorted(detected_poses.keys())
                known = sorted(marker_world_transforms.keys())
                print(
                    f"cam={cam_fps:4.1f}fps  mav={mav_hz:4.1f}Hz"
                    f"  NO POSE  visible={visible}  map={known}",
                    flush=True,
                )

            last_print_t = now


def main():
    # graceful Ctrl+C
    signal.signal(signal.SIGINT, lambda *_: sys.exit(0))

    if not CALIBRATION_JSON_PATH.exists():
        print(f"[Error] calibration file not found: {CALIBRATION_JSON_PATH}")
        sys.exit(1)
    if not MARKER_LAYOUT_JSON_PATH.exists():
        print(f"[Error] marker layout not found: {MARKER_LAYOUT_JSON_PATH}")
        sys.exit(1)

    camera_matrix, dist_coeffs = load_calibration()
    dictionary = get_dictionary()
    _, marker_world_transforms, marker_counts = load_marker_layout()

    print(f"[Map] loaded {len(marker_world_transforms)} markers: "
          f"{sorted(marker_world_transforms.keys())}")

    cap = open_camera()
    if cap is None:
        print(f"[Error] cannot open camera {CAMERA_SOURCE}")
        sys.exit(1)
    print(f"[Camera] opened: source={CAMERA_SOURCE} {TARGET_WIDTH}x{TARGET_HEIGHT}@{TARGET_FPS}")

    conn = connect_mavlink()

    try:
        run(cap, dictionary, camera_matrix, dist_coeffs, conn,
            marker_world_transforms, marker_counts)
    finally:
        cap.release()
        print("[Done]")


if __name__ == "__main__":
    main()
