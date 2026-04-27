"""Camera + ArUco pose test. Uses aruco_pose.ArucoPoseTracker — no detection logic here."""

import time

from aruco_pose import ArucoPoseTracker

FPS_UPDATE_PERIOD_SEC = 0.5
PRINT_PERIOD_SEC = 0.5


def main():
    tracker = None
    try:
        tracker = ArucoPoseTracker()
        print("Камера открыта. Ctrl+C — остановить.")

        fps_frame_count = 0
        fps_started_at = time.perf_counter()
        current_fps = 0.0
        last_print_at = 0.0
        last_status = None

        while True:
            pose = tracker.get_pose()

            fps_frame_count += 1
            now = time.perf_counter()
            elapsed = now - fps_started_at
            if elapsed >= FPS_UPDATE_PERIOD_SEC:
                current_fps = fps_frame_count / elapsed
                fps_frame_count = 0
                fps_started_at = now

            if pose is not None:
                x_m, y_m, z_m, yaw_deg = pose
                status = (
                    f"FPS={current_fps:5.1f} | "
                    f"X={x_m:8.3f}m  Y={y_m:8.3f}m  Z={z_m:8.3f}m | "
                    f"yaw={yaw_deg:8.2f}°"
                )
            else:
                status = f"FPS={current_fps:5.1f} | маркеры не найдены"

            if (now - last_print_at) >= PRINT_PERIOD_SEC or status != last_status:
                print(status)
                last_print_at = now
                last_status = status

    except KeyboardInterrupt:
        print("Остановлено.")
    finally:
        if tracker is not None:
            tracker.close()


if __name__ == "__main__":
    main()
