"""ArUco-based pose estimation library for rc_ch7.py."""

import json
import os
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent

CALIBRATION_JSON_PATH = SCRIPT_DIR / "camera_calibration.json"
MARKER_LAYOUT_JSON_PATH = SCRIPT_DIR / "marker_layout.json"

CAMERA_SOURCE = os.environ.get("ARUCO_CAMERA_SOURCE", "0")
CAMERA_BACKEND = os.environ.get("ARUCO_CAMERA_BACKEND", "auto").lower()
TARGET_WIDTH = int(os.environ.get("ARUCO_CAMERA_WIDTH", "1280"))
TARGET_HEIGHT = int(os.environ.get("ARUCO_CAMERA_HEIGHT", "720"))
TARGET_FPS = int(os.environ.get("ARUCO_CAMERA_FPS", "30"))
TARGET_FOURCC = os.environ.get("ARUCO_CAMERA_FOURCC", "MJPG")
LINUX_FALLBACK_SOURCES = ("/dev/video1", "/dev/video4", "/dev/video0")
_LAST_CAPTURE_INFO = None

_BACKEND_MAP = {
    "auto": cv2.CAP_ANY,
    "dshow": cv2.CAP_DSHOW,
    "msmf": getattr(cv2, "CAP_MSMF", cv2.CAP_ANY),
    "v4l2": cv2.CAP_V4L2,
    "gstreamer": getattr(cv2, "CAP_GSTREAMER", cv2.CAP_ANY),
}


def _load_calibration():
    payload = json.loads(CALIBRATION_JSON_PATH.read_text(encoding="utf-8"))
    camera_matrix = np.array(payload["camera_matrix"], dtype=np.float64)
    dist_coeffs = np.array(payload["dist_coeffs"], dtype=np.float64)
    return camera_matrix, dist_coeffs


def _load_marker_layout():
    payload = json.loads(MARKER_LAYOUT_JSON_PATH.read_text(encoding="utf-8"))
    dictionary_name = payload["dictionary_name"]
    marker_length_mm = float(payload["marker_length_mm"])

    marker_world_transforms = {}
    for marker_id_text, marker_payload in payload["markers"].items():
        marker_id = int(marker_id_text)
        rvec = np.array(marker_payload["rvec"], dtype=np.float64).reshape(3, 1)
        tvec = np.array(marker_payload["tvec"], dtype=np.float64).reshape(3, 1)
        marker_world_transforms[marker_id] = _rt_to_transform(rvec, tvec)

    return dictionary_name, marker_length_mm, marker_world_transforms


def _get_dictionary(dictionary_name: str):
    aruco = cv2.aruco
    return aruco.getPredefinedDictionary(getattr(aruco, dictionary_name))


class _ArucoDetectorCompat:
    """OpenCV ArUco detector wrapper for both old and new APIs."""

    def __init__(self, dictionary):
        self._dictionary = dictionary
        if hasattr(cv2.aruco, "ArucoDetector"):
            self._detector = cv2.aruco.ArucoDetector(dictionary)
            self._use_modern_api = True
        else:
            self._detector = None
            self._use_modern_api = False
            if hasattr(cv2.aruco, "DetectorParameters"):
                self._parameters = cv2.aruco.DetectorParameters()
            else:
                self._parameters = cv2.aruco.DetectorParameters_create()

    def detectMarkers(self, frame):
        if self._use_modern_api:
            return self._detector.detectMarkers(frame)
        return cv2.aruco.detectMarkers(frame, self._dictionary, parameters=self._parameters)


def _normalize_camera_source(source):
    if isinstance(source, int):
        return source
    source_text = str(source).strip()
    if source_text.isdigit():
        return int(source_text)
    return source_text


def _build_gstreamer_pipeline(source) -> Optional[str]:
    if not isinstance(source, str) or not source.startswith("/dev/video"):
        return None
    return (
        f"v4l2src device={source} io-mode=2 ! "
        f"image/jpeg,width={TARGET_WIDTH},height={TARGET_HEIGHT},framerate={TARGET_FPS}/1 ! "
        "jpegdec ! videoconvert ! appsink drop=true max-buffers=1 sync=false"
    )


def _camera_attempts(source):
    normalized_source = _normalize_camera_source(source)
    if CAMERA_BACKEND != "auto":
        return [(normalized_source, _BACKEND_MAP.get(CAMERA_BACKEND, cv2.CAP_ANY))]

    if os.name == "nt":
        return [
            (normalized_source, cv2.CAP_DSHOW),
            (normalized_source, getattr(cv2, "CAP_MSMF", cv2.CAP_ANY)),
            (normalized_source, cv2.CAP_ANY),
        ]

    attempts = []
    linux_sources = [normalized_source]
    if normalized_source == 0:
        for fallback_source in LINUX_FALLBACK_SOURCES:
            if fallback_source not in linux_sources:
                linux_sources.append(fallback_source)

    for linux_source in linux_sources:
        pipeline = _build_gstreamer_pipeline(linux_source)
        if pipeline is not None:
            attempts.append((pipeline, getattr(cv2, "CAP_GSTREAMER", cv2.CAP_ANY)))
            attempts.append((linux_source, cv2.CAP_V4L2))
        attempts.append((linux_source, cv2.CAP_ANY))
    return attempts


def _configure_camera(cap: cv2.VideoCapture) -> None:
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if TARGET_FOURCC:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*TARGET_FOURCC))
    if TARGET_WIDTH > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_WIDTH)
    if TARGET_HEIGHT > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_HEIGHT)
    if TARGET_FPS > 0:
        cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)


def _fourcc_to_str(value: float) -> str:
    code = int(value)
    if code <= 0:
        return "----"
    return "".join(chr((code >> (8 * index)) & 0xFF) for index in range(4))


def _describe_capture(cap: cv2.VideoCapture, source, backend: int) -> dict:
    return {
        "source": str(source),
        "backend": int(backend),
        "backend_name": cap.getBackendName() if hasattr(cap, "getBackendName") else str(backend),
        "requested_width": int(TARGET_WIDTH),
        "requested_height": int(TARGET_HEIGHT),
        "requested_fps": int(TARGET_FPS),
        "requested_fourcc": str(TARGET_FOURCC),
        "reported_width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "reported_height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "reported_fps": float(cap.get(cv2.CAP_PROP_FPS)),
        "reported_fourcc": _fourcc_to_str(cap.get(cv2.CAP_PROP_FOURCC)),
    }


def _open_camera() -> Optional[cv2.VideoCapture]:
    global _LAST_CAPTURE_INFO
    for source, backend in _camera_attempts(CAMERA_SOURCE):
        cap = cv2.VideoCapture(source, backend)
        if not cap.isOpened():
            cap.release()
            continue
        _configure_camera(cap)
        for _ in range(10):
            cap.read()
        ok, _ = cap.read()
        if ok:
            _LAST_CAPTURE_INFO = _describe_capture(cap, source, backend)
            return cap
        cap.release()
    return None


def _rt_to_transform(rvec, tvec) -> np.ndarray:
    rotation_matrix, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation_matrix
    transform[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
    return transform


def _invert_transform(transform: np.ndarray) -> np.ndarray:
    inverse = np.eye(4, dtype=np.float64)
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -(rotation.T @ translation)
    return inverse


def _rotation_matrix_to_euler_deg(rotation_matrix: np.ndarray):
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


def _rotation_matrix_to_quaternion(rotation_matrix: np.ndarray) -> np.ndarray:
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


def _quaternion_to_rotation_matrix(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(quaternion, dtype=np.float64)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _average_transforms(transforms):
    translations = np.array([t[:3, 3] for t in transforms], dtype=np.float64)
    quaternions = []
    reference_quaternion = None
    for t in transforms:
        q = _rotation_matrix_to_quaternion(t[:3, :3])
        if reference_quaternion is None:
            reference_quaternion = q
        elif np.dot(q, reference_quaternion) < 0.0:
            q = -q
        quaternions.append(q)
    mean_transform = np.eye(4, dtype=np.float64)
    mean_transform[:3, 3] = np.mean(translations, axis=0)
    mean_q = np.mean(np.array(quaternions, dtype=np.float64), axis=0)
    mean_q /= np.linalg.norm(mean_q)
    mean_transform[:3, :3] = _quaternion_to_rotation_matrix(mean_q)
    return mean_transform


def _solve_marker_pose(marker_corners, camera_matrix, dist_coeffs, marker_length_mm):
    half = marker_length_mm * 0.5
    object_points = np.array(
        [[-half, half, 0.0], [half, half, 0.0], [half, -half, 0.0], [-half, -half, 0.0]],
        dtype=np.float32,
    )
    image_points = np.asarray(marker_corners, dtype=np.float32).reshape(4, 2)
    ok, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    if not ok:
        return False, None, None
    return True, rvec, tvec


def _compute_reprojection_error(
    marker_corners,
    rvec,
    tvec,
    camera_matrix,
    dist_coeffs,
    marker_length_mm,
) -> float:
    half = marker_length_mm * 0.5
    object_points = np.array(
        [[-half, half, 0.0], [half, half, 0.0], [half, -half, 0.0], [-half, -half, 0.0]],
        dtype=np.float32,
    )
    image_points = np.asarray(marker_corners, dtype=np.float32).reshape(4, 2)
    projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs)
    return float(np.mean(np.linalg.norm(image_points - projected.reshape(4, 2), axis=1)))


def _detect_marker_poses(
    frame,
    detector,
    camera_matrix,
    dist_coeffs,
    marker_length_mm,
):
    marker_corners, marker_ids, _ = detector.detectMarkers(frame)
    poses = {}

    if marker_ids is None or len(marker_ids) == 0:
        return marker_corners, marker_ids, poses

    for corners, marker_id in zip(marker_corners, marker_ids.flatten()):
        ok, rvec, tvec = _solve_marker_pose(corners, camera_matrix, dist_coeffs, marker_length_mm)
        if not ok:
            continue
        marker_id = int(marker_id)
        poses[marker_id] = {
            "corners": corners,
            "rvec": rvec,
            "tvec": tvec,
            "camera_from_marker": _rt_to_transform(rvec, tvec),
            "reprojection_error": _compute_reprojection_error(
                corners,
                rvec,
                tvec,
                camera_matrix,
                dist_coeffs,
                marker_length_mm,
            ),
        }

    return marker_corners, marker_ids, poses


def _compute_pose_consistency(transforms) -> Tuple[Optional[float], Optional[float]]:
    if len(transforms) < 2:
        return None, None
    positions = np.array([t[:3, 3] for t in transforms], dtype=np.float64)
    trans_mm = float(np.mean(np.std(positions, axis=0)))
    quaternions = []
    ref_q = None
    for t in transforms:
        q = _rotation_matrix_to_quaternion(t[:3, :3])
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


def _detect_and_estimate(
    frame,
    detector,
    camera_matrix,
    dist_coeffs,
    marker_length_mm,
    marker_world_transforms,
):
    _, _, detected_poses = _detect_marker_poses(
        frame,
        detector,
        camera_matrix,
        dist_coeffs,
        marker_length_mm,
    )
    world_from_camera_candidates = []
    for marker_id, pose in detected_poses.items():
        if marker_id not in marker_world_transforms:
            continue
        world_from_camera_candidates.append(
            marker_world_transforms[marker_id] @ _invert_transform(pose["camera_from_marker"])
        )

    if not world_from_camera_candidates:
        return None

    return _average_transforms(world_from_camera_candidates)


class ArucoPoseTracker:
    """Opens camera, loads calibration and marker map, estimates pose per frame."""

    def __init__(self):
        if not CALIBRATION_JSON_PATH.exists():
            raise FileNotFoundError(f"Calibration file not found: {CALIBRATION_JSON_PATH}")
        if not MARKER_LAYOUT_JSON_PATH.exists():
            raise FileNotFoundError(f"Marker layout file not found: {MARKER_LAYOUT_JSON_PATH}")

        self._camera_matrix, self._dist_coeffs = _load_calibration()
        dictionary_name, self._marker_length_mm, self._marker_world_transforms = _load_marker_layout()
        self._dictionary = _get_dictionary(dictionary_name)
        self._detector = _ArucoDetectorCompat(self._dictionary)
        self._cap = _open_camera()
        if self._cap is None:
            raise RuntimeError(f"Cannot open camera source {CAMERA_SOURCE}")
        self._capture_info = dict(_LAST_CAPTURE_INFO or {})

    def get_pose(self) -> Optional[Tuple[float, float, float, float]]:
        """Read one frame and return (x_m, y_m, z_m, yaw_deg) or None if no known markers visible."""
        ok, frame = self._cap.read()
        if not ok:
            return None

        world_from_camera = _detect_and_estimate(
            frame,
            self._detector,
            self._camera_matrix,
            self._dist_coeffs,
            self._marker_length_mm,
            self._marker_world_transforms,
        )
        if world_from_camera is None:
            return None

        xyz_mm = world_from_camera[:3, 3]
        rpy_deg = _rotation_matrix_to_euler_deg(world_from_camera[:3, :3])

        x_m = float(xyz_mm[0]) / 1000.0
        y_m = float(xyz_mm[1]) / 1000.0
        z_m = float(xyz_mm[2]) / 1000.0
        yaw_deg = -float(rpy_deg[2])

        return x_m, y_m, z_m, yaw_deg

    def get_capture_info(self) -> dict:
        return dict(self._capture_info or {})

    def analyze_frame(self):
        ok, frame = self._cap.read()
        if not ok:
            return None

        marker_corners, marker_ids, detected_poses = _detect_marker_poses(
            frame,
            self._detector,
            self._camera_matrix,
            self._dist_coeffs,
            self._marker_length_mm,
        )

        world_from_camera_candidates = []
        used_marker_ids = []
        for marker_id, pose in detected_poses.items():
            if marker_id not in self._marker_world_transforms:
                continue
            world_from_camera_candidates.append(
                self._marker_world_transforms[marker_id] @ _invert_transform(pose["camera_from_marker"])
            )
            used_marker_ids.append(marker_id)

        world_from_camera = (
            _average_transforms(world_from_camera_candidates)
            if world_from_camera_candidates
            else None
        )
        consistency_mm, consistency_deg = _compute_pose_consistency(world_from_camera_candidates)
        return {
            "frame": frame,
            "marker_corners": marker_corners,
            "marker_ids": marker_ids,
            "detected_poses": detected_poses,
            "world_from_camera": world_from_camera,
            "used_marker_ids": used_marker_ids,
            "consistency_mm": consistency_mm,
            "consistency_deg": consistency_deg,
            "capture_info": self.get_capture_info(),
            "marker_layout_ids": sorted(self._marker_world_transforms),
            "marker_length_mm": float(self._marker_length_mm),
        }

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
