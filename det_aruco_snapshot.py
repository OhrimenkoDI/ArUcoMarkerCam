"""Save a diagnostic det_aruco snapshot with marker and pose overlay."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from aruco_pose import ArucoPoseTracker, _rotation_matrix_to_euler_deg

TEXT_COLOR = (0, 255, 0)
INFO_COLOR = (255, 220, 0)
MARKER_COLOR = (0, 200, 255)
ERROR_COLOR = (80, 80, 255)
QUALITY_GOOD_COLOR = (0, 220, 0)
QUALITY_MED_COLOR = (0, 200, 220)
QUALITY_BAD_COLOR = (60, 60, 255)
AXIS_LENGTH_MM = 60.0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Capture one annotated frame using the same pose logic as det_aruco.py.",
    )
    parser.add_argument(
        "--delay-sec",
        type=float,
        default=1.0,
        help="How long to wait before saving the frame.",
    )
    parser.add_argument(
        "--output",
        default="debug_det_aruco_snapshot.png",
        help="Output image path.",
    )
    parser.add_argument(
        "--meta-output",
        default="debug_det_aruco_snapshot.json",
        help="Output metadata JSON path.",
    )
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=0,
        help="Extra frames to discard after the delay before saving.",
    )
    return parser.parse_args()


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


def draw_world_pose_text(frame, world_from_camera, x, y, color):
    position = world_from_camera[:3, 3]
    angles_deg = _rotation_matrix_to_euler_deg(world_from_camera[:3, :3])
    draw_multiline_text(
        frame,
        [
            "Camera pose in marker map",
            f"World XYZ mm: {position[0]:7.1f} {position[1]:7.1f} {position[2]:7.1f}",
            f"World RPY deg: {angles_deg[0]:6.1f} {angles_deg[1]:6.1f} {angles_deg[2]:6.1f}",
        ],
        x,
        y,
        color,
    )


def draw_quality_score(frame, consistency_mm, x, y):
    if consistency_mm is None:
        label = "Map quality: --- (need 2+ markers)"
        color = INFO_COLOR
    elif consistency_mm < 5.0:
        label = f"Map quality: {consistency_mm:.1f} mm  GOOD"
        color = QUALITY_GOOD_COLOR
    elif consistency_mm < 20.0:
        label = f"Map quality: {consistency_mm:.1f} mm  OK"
        color = QUALITY_MED_COLOR
    else:
        label = f"Map quality: {consistency_mm:.1f} mm  POOR"
        color = QUALITY_BAD_COLOR
    cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)


def draw_marker_visuals(frame, marker_corners, marker_ids, detected_poses, camera_matrix, dist_coeffs):
    if marker_ids is None or len(marker_ids) == 0:
        return

    cv2.aruco.drawDetectedMarkers(frame, marker_corners, marker_ids, MARKER_COLOR)
    for marker_id in marker_ids.flatten():
        marker_id = int(marker_id)
        pose = detected_poses.get(marker_id)
        if pose is None:
            continue

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
        cv2.putText(
            frame,
            f"id={marker_id}",
            (anchor[0], top_y - 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            MARKER_COLOR,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"e={pose['reprojection_error']:.1f}px",
            (anchor[0], top_y - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            ERROR_COLOR,
            2,
            cv2.LINE_AA,
        )


def render_overlay(tracker: ArucoPoseTracker, analysis: dict, delay_sec: float):
    frame = analysis["frame"].copy()
    draw_marker_visuals(
        frame,
        analysis["marker_corners"],
        analysis["marker_ids"],
        analysis["detected_poses"],
        tracker._camera_matrix,
        tracker._dist_coeffs,
    )

    if analysis["world_from_camera"] is not None:
        draw_world_pose_text(frame, analysis["world_from_camera"], 10, 65, INFO_COLOR)

    draw_quality_score(frame, analysis["consistency_mm"], 10, 360)

    capture_info = analysis["capture_info"]
    used_marker_ids = sorted(set(analysis["used_marker_ids"]))
    draw_multiline_text(
        frame,
        [
            "det_aruco diagnostic snapshot",
            f"Map markers: {', '.join(str(v) for v in analysis['marker_layout_ids'])}",
            f"Used markers: {', '.join(str(v) for v in used_marker_ids) if used_marker_ids else '---'}",
            f"Requested: {capture_info.get('requested_width')}x{capture_info.get('requested_height')} @ "
            f"{capture_info.get('requested_fps')} {capture_info.get('requested_fourcc')}",
            f"Reported: {capture_info.get('reported_width')}x{capture_info.get('reported_height')} @ "
            f"{capture_info.get('reported_fps', 0.0):.2f} {capture_info.get('reported_fourcc')}",
            f"Source: {capture_info.get('source')} backend={capture_info.get('backend_name')}",
            f"Delay before shot: {delay_sec:.2f}s",
        ],
        10,
        390,
        INFO_COLOR,
    )
    cv2.putText(
        frame,
        f"Visible markers: {0 if analysis['marker_ids'] is None else len(analysis['marker_ids'])}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        TEXT_COLOR,
        2,
        cv2.LINE_AA,
    )
    return frame


def build_metadata(analysis: dict, image_path: Path, delay_sec: float) -> dict:
    world_from_camera = analysis["world_from_camera"]
    if world_from_camera is None:
        pose_payload = None
    else:
        pose_payload = {
            "xyz_mm": world_from_camera[:3, 3].tolist(),
            "rpy_deg": _rotation_matrix_to_euler_deg(world_from_camera[:3, :3]).tolist(),
        }

    marker_payloads = []
    for marker_id in sorted(analysis["detected_poses"]):
        pose = analysis["detected_poses"][marker_id]
        marker_payloads.append(
            {
                "id": int(marker_id),
                "reprojection_error_px": float(pose["reprojection_error"]),
                "tvec_mm": np.asarray(pose["tvec"], dtype=np.float64).reshape(3).tolist(),
                "rvec": np.asarray(pose["rvec"], dtype=np.float64).reshape(3).tolist(),
            }
        )

    return {
        "saved_image": str(image_path),
        "delay_sec": float(delay_sec),
        "capture_info": analysis["capture_info"],
        "visible_marker_ids": [] if analysis["marker_ids"] is None else [int(v) for v in analysis["marker_ids"].flatten()],
        "used_marker_ids": [int(v) for v in sorted(set(analysis["used_marker_ids"]))],
        "consistency_mm": analysis["consistency_mm"],
        "world_pose": pose_payload,
        "markers": marker_payloads,
    }


def main():
    args = parse_args()
    output_path = Path(args.output)
    meta_output_path = Path(args.meta_output)

    tracker = None
    try:
        tracker = ArucoPoseTracker()
        started_at = time.perf_counter()
        while (time.perf_counter() - started_at) < args.delay_sec:
            analysis = tracker.analyze_frame()
            if analysis is None:
                raise RuntimeError("Failed to read frame during delay period.")

        analysis = None
        for _ in range(max(args.warmup_frames, 0) + 1):
            analysis = tracker.analyze_frame()
            if analysis is None:
                raise RuntimeError("Failed to read frame for snapshot.")

        overlay = render_overlay(tracker, analysis, args.delay_sec)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        meta_output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), overlay)
        meta_output_path.write_text(
            json.dumps(build_metadata(analysis, output_path, args.delay_sec), indent=2),
            encoding="utf-8",
        )
        print(f"Saved overlay image to {output_path}")
        print(f"Saved metadata JSON to {meta_output_path}")
    finally:
        if tracker is not None:
            tracker.close()


if __name__ == "__main__":
    main()
