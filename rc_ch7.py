import math
import sys
import threading
import time

from pymavlink import mavutil

from aruco_pose import ArucoPoseTracker

# ── Выбор интерфейса ──────────────────────────────────────────────────────────
# Установи одну из констант в True, вторую в False

USE_UDP  = False    # UDP: подключение через MAVProxy (--out=udpin:0.0.0.0:14552)
USE_UART = True   # UART: прямое подключение к полётному контроллеру по COM-порту

# ── Настройки UDP ─────────────────────────────────────────────────────────────
UDP_IP   = "127.0.0.1"   #"10.0.20.15"
UDP_PORT = 14552         #14550

# ── Настройки UART ────────────────────────────────────────────────────────────
# Orange Pi 5 Ultra: UART3_M1 → /dev/ttyS3  (pins 8=TX, 10=RX)
# USB-UART адаптер:  /dev/ttyUSB0
UART_PORT = "/dev/ttyS3"
UART_BAUD = 921600

# ── RC каналы ─────────────────────────────────────────────────────────────────
# CH7 values for GPS source selection (RC7_OPTION=90 in ArduPilot)
# 1000 = Source 1 (primary GPS)
# 1500 = Source 2
# 2000 = Source 3
PWM_CH6 = 2000
PWM_CH7 = 1500

INTERVAL = 0.02  # 50 Hz — держит override активным (RC_OVERRIDE_TIME = 3 сек)

# ── Внешний азимут (вместо компаса) ──────────────────────────────────────────
AZIMUTH_DEG = 180  # резервное значение, если маркеры не видны
# ─────────────────────────────────────────────────────────────────────────────


CAMERA_AZIMUTH_OFFSET_DEG = 180.0  # camera yaw correction: final yaw = measured yaw + this constant
ALPHA_POS = 0.30
ALPHA_YAW = 0.20
MAX_POS_JUMP_M = 0.45
MAX_YAW_JUMP_DEG = 10.0
MAX_REJECTS_BEFORE_RESET = 10
POSE_TIMEOUT_SEC = 1.0


def pose_to_ardupilot_position(x_m: float, y_m: float, z_m: float):
    """Convert marker-map pose to ArduPilot local position: X forward, Y right, Z down."""
    return y_m, x_m, -z_m


def angle_diff_deg(a: float, b: float) -> float:
    return (a - b + 180.0) % 360.0 - 180.0


class EMAFilter:
    def __init__(self, alpha_pos: float = ALPHA_POS, alpha_yaw: float = ALPHA_YAW):
        self._alpha_pos = alpha_pos
        self._alpha_yaw = alpha_yaw
        self._x = None
        self._y = None
        self._z = None
        self._yaw = None

    @property
    def initialised(self) -> bool:
        return self._x is not None

    def update(self, x_m: float, y_m: float, z_m: float, yaw_deg: float):
        if not self.initialised:
            self._x, self._y, self._z, self._yaw = x_m, y_m, z_m, yaw_deg
        else:
            a = self._alpha_pos
            self._x = a * x_m + (1.0 - a) * self._x
            self._y = a * y_m + (1.0 - a) * self._y
            self._z = a * z_m + (1.0 - a) * self._z
            self._yaw = (self._yaw + self._alpha_yaw * angle_diff_deg(yaw_deg, self._yaw)) % 360.0
        return self.get()

    def get(self):
        if not self.initialised:
            return None
        return self._x, self._y, self._z, self._yaw

    def reset(self) -> None:
        self._x = None
        self._y = None
        self._z = None
        self._yaw = None


class PoseWorker:
    def __init__(self, tracker: ArucoPoseTracker):
        self._tracker = tracker
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, name="aruco-pose", daemon=True)
        self._pose = None
        self._visible = False
        self._frames = 0
        self._error = None

    def start(self) -> None:
        self._thread.start()

    def snapshot(self):
        with self._lock:
            return self._pose, self._visible, self._frames, self._error

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                pose = self._tracker.get_pose()
            except Exception as exc:
                with self._lock:
                    self._error = exc
                return

            with self._lock:
                self._frames += 1
                self._visible = pose is not None
                self._pose = pose


def make_connection() -> mavutil.mavfile:
    if USE_UDP and not USE_UART:
        addr = f"udpout:{UDP_IP}:{UDP_PORT}"
        print(f"UDP  -> {UDP_IP}:{UDP_PORT}")
    elif USE_UART and not USE_UDP:
        addr = f"{UART_PORT},{UART_BAUD}"
        print(f"UART -> {UART_PORT}  {UART_BAUD} baud")
    else:
        print("Ошибка: установи ровно одну константу USE_UDP или USE_UART в True", file=sys.stderr)
        sys.exit(1)

    return mavutil.mavlink_connection(addr, force_connected=True)


def send_heading0(conn, azimuth_deg: float) -> None:
    """Инжектирует внешний azimuth в EKF3 через VISION_POSITION_ESTIMATE.
    Заменяет компас; yaw в радианах, NED-фрейм."""
    yaw_rad = math.radians(azimuth_deg)
    usec = int(time.time() * 1e6)
    conn.mav.vision_position_estimate_send(
        usec,
        0.0, 0.0, 0.0,   # x, y, z — позицию не передаём
        0.0, 0.0, yaw_rad,  # roll, pitch, yaw
    )

def send_heading(conn, azimuth_deg: float,
                 x: float = 0.1, y: float = 0.3, z: float = -1.6) -> None:
    """
    Передаёт позицию + yaw через ATT_POS_MOCAP (#138).
    EK3_SRC1_YAW=6 + VISO_TYPE=1 читают именно этот message.
    Координаты в NED-фрейме, метры.
    """
    yaw_rad = math.radians(azimuth_deg)

    # Quaternion: roll=0, pitch=0, yaw=az
    q = [
        math.cos(yaw_rad / 2),  # w
        0.0,                     # x
        0.0,                     # y
        math.sin(yaw_rad / 2),  # z
    ]

    usec = int(time.time() * 1e6)
    conn.mav.att_pos_mocap_send(
        usec,
        q,
        x, y, z,
    )



def send_override(conn, pwm6: int, pwm7: int) -> None:
    conn.mav.rc_channels_override_send(
        1, 1,             # target_system, target_component
        0, 0, 0, 0, 0,   # CH1-CH5: 0 = не переопределять
        pwm6,             # CH6
        pwm7,             # CH7
        0,                # CH8
    )

def send_gps_origin(conn, lat_deg: float, lon_deg: float, alt_m: float = 0.0) -> None:
    """Устанавливает глобальный origin для EKF3. Нужен для отображения на карте."""
    conn.mav.set_gps_global_origin_send(
        1,                          # target_system
        int(lat_deg * 1e7),         # latitude  (градусы × 1e7)
        int(lon_deg * 1e7),         # longitude (градусы × 1e7)
        int(alt_m * 1000),          # altitude  (мм)
    )


def main() -> None:
    conn = make_connection()
    print(f"RC_OVERRIDE  CH6={PWM_CH6}  CH7={PWM_CH7}  AZIMUTH={AZIMUTH_DEG}° (резерв)")
    print("Ctrl+C — остановить и отпустить каналы")

    # Задаём origin — реальные координаты твоей точки старта
    send_gps_origin(conn, lat_deg=60.00, lon_deg=30.0, alt_m=150.0)
    print("GPS origin установлен")

    tracker = None
    pose_worker = None
    try:
        tracker = ArucoPoseTracker()
        print("ArUco трекер запущен")
        print(f"Capture: {tracker.get_capture_info()}")

        pose_worker = PoseWorker(tracker)
        pose_worker.start()

        sent = 0
        last_rate_time = time.perf_counter()
        last_vision_frames = 0
        ema = EMAFilter()
        prev_raw_xy = None
        prev_yaw_deg = None
        reject_count = 0
        last_pose = None
        last_pose_time = -999.0
        frozen_yaw_deg = float(AZIMUTH_DEG)
        while True:
            t0 = time.perf_counter()

            pose, marker_visible, vision_frames, vision_error = pose_worker.snapshot()
            if vision_error is not None:
                raise vision_error

            if pose is not None:
                x_raw, y_raw, z_raw, yaw_raw = pose
                yaw_adjusted = (yaw_raw + CAMERA_AZIMUTH_OFFSET_DEG) % 360.0

                stale_pose = (time.perf_counter() - last_pose_time) >= POSE_TIMEOUT_SEC
                if stale_pose:
                    ema.reset()
                    prev_raw_xy = None
                    prev_yaw_deg = None
                    reject_count = 0

                pos_ok = True
                if prev_raw_xy is not None:
                    jump_m = math.hypot(x_raw - prev_raw_xy[0], y_raw - prev_raw_xy[1])
                    if jump_m > MAX_POS_JUMP_M:
                        pos_ok = False
                        print(f"  [SKIP] pos jump {jump_m * 100:.1f} cm > {MAX_POS_JUMP_M * 100:.0f} cm")

                yaw_ok = True
                if prev_yaw_deg is not None:
                    yaw_jump = abs(angle_diff_deg(yaw_adjusted, prev_yaw_deg))
                    if yaw_jump > MAX_YAW_JUMP_DEG:
                        yaw_ok = False
                        print(f"  [SKIP] yaw jump {yaw_jump:.1f}deg > {MAX_YAW_JUMP_DEG:.0f}deg")

                if pos_ok and yaw_ok:
                    prev_raw_xy = (x_raw, y_raw)
                    prev_yaw_deg = yaw_adjusted
                    last_pose = ema.update(x_raw, y_raw, z_raw, yaw_adjusted)
                    last_pose_time = time.perf_counter()
                    frozen_yaw_deg = last_pose[3]
                    reject_count = 0
                else:
                    reject_count += 1
                    if reject_count >= MAX_REJECTS_BEFORE_RESET:
                        print("  [RESET] too many rejected poses, accepting next measurement as new baseline")
                        ema.reset()
                        prev_raw_xy = None
                        prev_yaw_deg = None
                        last_pose = None
                        last_pose_time = -999.0
                        reject_count = 0

            pose_age = time.perf_counter() - last_pose_time
            if last_pose is not None and pose_age < POSE_TIMEOUT_SEC:
                x_m, y_m, z_m, yaw_deg = last_pose
                src = "aruco" if marker_visible else f"last({pose_age:.1f}s)"
            else:
                x_m, y_m, z_m = 0.0, 0.0, 0.0
                yaw_deg = frozen_yaw_deg
                src = "резерв"

            yaw_step_deg = None
            if prev_yaw_deg is not None:
                yaw_step_deg = abs(angle_diff_deg(yaw_deg, prev_yaw_deg))

            ap_x_m, ap_y_m, ap_z_m = pose_to_ardupilot_position(x_m, y_m, z_m)

            #send_override(conn, PWM_CH6, PWM_CH7)
            send_heading(conn, yaw_deg, ap_x_m, ap_y_m, ap_z_m)
            sent += 1
            if sent % 50 == 0:
                now = time.perf_counter()
                dt = now - last_rate_time
                real_hz = 50.0 / dt if dt > 0 else 0.0
                vision_hz = (vision_frames - last_vision_frames) / dt if dt > 0 else 0.0
                last_rate_time = now
                last_vision_frames = vision_frames
                step_info = ""
                highlight_on = ""
                highlight_off = ""
                if yaw_step_deg is not None:
                    step_info = f"  d_az={yaw_step_deg:.1f}deg"
                    if yaw_step_deg > 5.0:
                        step_info += "  !!! angle step > 5deg"
                        highlight_on = "\033[93m"
                        highlight_off = "\033[0m"
                print(f"{highlight_on}  sent={sent}  send_hz={real_hz:.1f}  vision_hz={vision_hz:.1f}  [{src}]  aruco_x={x_m:.3f}m  aruco_y={y_m:.3f}m  aruco_z={z_m:.3f}m  ap_x={ap_x_m:.3f}m  ap_y={ap_y_m:.3f}m  ap_z={ap_z_m:.3f}m  az={yaw_deg:.1f}deg{step_info}{highlight_off}")

            remaining = INTERVAL - (time.perf_counter() - t0)
            if remaining > 0:
                time.sleep(remaining)

    except KeyboardInterrupt:
        print("\nОтпускаю каналы...")
        for _ in range(5):
            send_override(conn, 0, 0)
            time.sleep(INTERVAL)

    finally:
        if pose_worker is not None:
            pose_worker.close()
        if tracker is not None:
            tracker.close()
        conn.close()
        print("Закрыто.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
