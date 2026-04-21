import os
import time

import cv2

OUTPUT_DIR = "images"
CAMERA_SOURCE = 0
CAMERA_BACKEND = "auto"
TARGET_WIDTH = 1280
TARGET_HEIGHT = 720
TARGET_FPS = 30
TARGET_FOURCC = "MJPG"
MAX_IMAGES = 100
SAVE_DELAY_SEC = 0.3
WINDOW_TITLE = "Camera Capture - SPACE: save, ESC: exit"

BACKEND_MAP = {
    "auto": cv2.CAP_ANY,
    "dshow": cv2.CAP_DSHOW,
    "v4l2": cv2.CAP_V4L2,
}


def fourcc_to_str(value):
    code = int(value)
    if code <= 0:
        return "N/A"
    return "".join(chr((code >> (8 * i)) & 0xFF) for i in range(4))


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


def print_mode_info(cap, first_frame):
    reported_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    reported_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    reported_fps = cap.get(cv2.CAP_PROP_FPS)
    reported_fourcc = fourcc_to_str(cap.get(cv2.CAP_PROP_FOURCC))

    frame_height, frame_width = first_frame.shape[:2]
    actual_width = reported_width if reported_width > 0 else frame_width
    actual_height = reported_height if reported_height > 0 else frame_height

    print("Capture settings:")
    print(f"  source      : {CAMERA_SOURCE}")
    print(f"  backend     : {CAMERA_BACKEND}")
    print(
        f"  request     : {TARGET_WIDTH}x{TARGET_HEIGHT}@{TARGET_FPS} {TARGET_FOURCC}"
    )
    print(
        f"  actual      : {actual_width}x{actual_height}, "
        f"fps={reported_fps:.2f}, fourcc={reported_fourcc}"
    )


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    cap = open_camera()
    if cap is None:
        print(f"Error: cannot open camera source {CAMERA_SOURCE}")
        return

    ok, frame = cap.read()
    if not ok:
        print("Error: first frame read failed")
        cap.release()
        return

    print_mode_info(cap, frame)
    print("\nPress SPACE to save image, ESC to exit.")

    saved_count = 0
    while saved_count < MAX_IMAGES:
        ok, frame = cap.read()
        if not ok:
            print("Frame read error")
            break

        preview = frame.copy()
        cv2.putText(
            preview,
            f"Saved: {saved_count}/{MAX_IMAGES}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow(WINDOW_TITLE, preview)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            print(f"\nCapture finished. Saved {saved_count} images.")
            break
        if key == 32:
            filename = os.path.join(OUTPUT_DIR, f"frame_{saved_count:03d}.jpg")
            cv2.imwrite(filename, frame)
            print(f"Saved: {filename}")
            saved_count += 1
            time.sleep(SAVE_DELAY_SEC)

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nTotal saved: {saved_count} images in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
