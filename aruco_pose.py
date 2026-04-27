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
    if isinstance(normalized_source, str) and normalized_source.startswith("/dev/video"):
        attempts.append((normalized_source, cv2.CAP_V4L2))
    attempts.append((normalized_source, cv2.CAP_ANY))
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


def _open_camera() -> Optional[cv2.VideoCapture]:
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


def _average_transforms(transforms):
    translations = np.array([t[:3, 3] for t in transforms], dtype=np.float64)
    rotvecs = []
    for t in transforms:
        rvec, _ = cv2.Rodrigues(t[:3, :3])
        rotvecs.append(rvec.reshape(3))
    mean_transform = np.eye(4, dtype=np.float64)
    mean_transform[:3, 3] = np.mean(translations, axis=0)
    mean_rvec = np.mean(np.array(rotvecs, dtype=np.float64), axis=0).reshape(3, 1)
    mean_transform[:3, :3], _ = cv2.Rodrigues(mean_rvec)
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


def _detect_and_estimate(
    frame,
    detector,
    camera_matrix,
    dist_coeffs,
    marker_length_mm,
    marker_world_transforms,
):
    marker_corners, marker_ids, _ = detector.detectMarkers(frame)

    if marker_ids is None or len(marker_ids) == 0:
        return None

    world_from_camera_candidates = []
    for corners, marker_id in zip(marker_corners, marker_ids.flatten()):
        marker_id = int(marker_id)
        if marker_id not in marker_world_transforms:
            continue
        ok, rvec, tvec = _solve_marker_pose(corners, camera_matrix, dist_coeffs, marker_length_mm)
        if not ok:
            continue
        camera_from_marker = _rt_to_transform(rvec, tvec)
        world_from_camera_candidates.append(
            marker_world_transforms[marker_id] @ _invert_transform(camera_from_marker)
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

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
