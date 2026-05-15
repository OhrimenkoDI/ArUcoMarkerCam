"""
rsV2.py — ArduPilot visual navigation via ArUco + EKF3
Улучшения по сравнению с rc_ch7.py:
  - EMA-фильтр позиции и yaw (без прыжков через 0/360)
  - Отбраковка резких скачков позиции (> MAX_POS_JUMP_M)
  - Таймаут last_pose (> POSE_TIMEOUT_SEC → не отправляем)
  - Явная ковариация в ATT_POS_MOCAP
  - Лимит на прыжок yaw (> MAX_YAW_JUMP_DEG → скип итерации)
"""

import math
import sys
import threading
import time

from pymavlink import mavutil

from aruco_pose import ArucoPoseTracker

# ── Выбор интерфейса ──────────────────────────────────────────────────────────
USE_UDP  = False   # UDP через MAVProxy  --out=udpin:0.0.0.0:14552
USE_UART = True    # UART прямо на FC

# ── Настройки UDP ─────────────────────────────────────────────────────────────
UDP_IP   = "127.0.0.1"
UDP_PORT = 14552

# ── Настройки UART ────────────────────────────────────────────────────────────
# Orange Pi 5 Ultra: UART3_M1 → /dev/ttyS3  (pin 8=TX, pin 10=RX)
UART_PORT = "/dev/ttyS3"
UART_BAUD = 921600

# ── RC каналы (если нужны) ────────────────────────────────────────────────────
PWM_CH6 = 2000
PWM_CH7 = 1500

INTERVAL = 0.02   # 50 Гц

# ── Внешний азимут (резерв, если маркеры не видны) ───────────────────────────
AZIMUTH_DEG = 180.0

# ── Коррекция угла камеры ─────────────────────────────────────────────────────
CAMERA_AZIMUTH_OFFSET_DEG = 180.0

# ── Параметры EMA-фильтра ─────────────────────────────────────────────────────
ALPHA_POS = 0.30   # 0 < α ≤ 1; меньше → плавнее, но больше лаг
ALPHA_YAW = 0.20

# ── Защита от выбросов ────────────────────────────────────────────────────────
MAX_POS_JUMP_M   = 0.15   # максимальный прыжок XY за один кадр, м
MAX_YAW_JUMP_DEG = 10.0   # максимальный прыжок yaw за один кадр, градус
MAX_REJECTS_BEFORE_RESET = 10  # после серии скипов переинициализировать фильтр

# ── Таймаут последней известной позы ─────────────────────────────────────────
POSE_TIMEOUT_SEC = 1.0    # если маркеры не видны дольше — не отправлять

# ── Ковариация для ATT_POS_MOCAP ──────────────────────────────────────────────
# Верхнетреугольная матрица 6×6 (21 элемент); единицы: м², рад²
# [0]=Xvar [6]=Yvar [11]=Zvar [15]=rollVar [18]=pitchVar [20]=yawVar
COVARIANCE = [
    0.01, 0, 0, 0, 0, 0,
          0.01, 0, 0, 0, 0,
                0.01, 0, 0, 0,
                      0.01, 0, 0,
                            0.01, 0,
                                  0.05,
]

_ATT_POS_MOCAP_SUPPORTS_COVARIANCE = True

# ─────────────────────────────────────────────────────────────────────────────


# ╔══════════════════════════════════════════════════════════════════════════════
# ║  EMA-фильтр
# ╚══════════════════════════════════════════════════════════════════════════════

class EMAFilter:
    """Экспоненциальное скользящее среднее для (x, y, z, yaw).
    Yaw обрабатывается через sin/cos — нет прыжка через 0°/360°.
    """

    def __init__(self, alpha_pos: float = ALPHA_POS, alpha_yaw: float = ALPHA_YAW):
        self._alpha_pos = alpha_pos
        self._alpha_yaw = alpha_yaw
        self._x: float | None = None
        self._y: float | None = None
        self._z: float | None = None
        self._yaw: float | None = None

    @property
    def initialised(self) -> bool:
        return self._x is not None

    def update(self, x: float, y: float, z: float, yaw_deg: float):
        """Принять новое измерение и вернуть сглаженное (x, y, z, yaw_deg)."""
        if not self.initialised:
            self._x, self._y, self._z, self._yaw = x, y, z, yaw_deg
        else:
            a = self._alpha_pos
            self._x = a * x + (1 - a) * self._x
            self._y = a * y + (1 - a) * self._y
            self._z = a * z + (1 - a) * self._z

            # Сглаживание yaw через кратчайшую дугу
            ay = self._alpha_yaw
            delta = _angle_diff_deg(yaw_deg, self._yaw)
            self._yaw = (self._yaw + ay * delta) % 360.0

        return self._x, self._y, self._z, self._yaw

    def get(self):
        if not self.initialised:
            return None
        return self._x, self._y, self._z, self._yaw

    def reset(self) -> None:
        self._x = None
        self._y = None
        self._z = None
        self._yaw = None


def _angle_diff_deg(a: float, b: float) -> float:
    """Кратчайшая разница углов (−180…+180)."""
    return (a - b + 180.0) % 360.0 - 180.0


# ╔══════════════════════════════════════════════════════════════════════════════
# ║  Поток захвата позы с камеры
# ╚══════════════════════════════════════════════════════════════════════════════

class PoseWorker:
    def __init__(self, tracker: ArucoPoseTracker):
        self._tracker = tracker
        self._stop    = threading.Event()
        self._lock    = threading.Lock()
        self._thread  = threading.Thread(target=self._run, name="aruco-pose", daemon=True)
        self._pose    = None
        self._visible = False
        self._frames  = 0
        self._error   = None

    def start(self) -> None:
        self._thread.start()

    def snapshot(self):
        """Вернуть (pose | None, marker_visible, total_frames, error | None)."""
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


# ╔══════════════════════════════════════════════════════════════════════════════
# ║  Вспомогательные функции MAVLink
# ╚══════════════════════════════════════════════════════════════════════════════

def make_connection() -> mavutil.mavfile:
    if USE_UDP and not USE_UART:
        addr = f"udpout:{UDP_IP}:{UDP_PORT}"
        print(f"UDP  -> {UDP_IP}:{UDP_PORT}")
    elif USE_UART and not USE_UDP:
        addr = f"{UART_PORT},{UART_BAUD}"
        print(f"UART -> {UART_PORT}  {UART_BAUD} baud")
    else:
        print("Ошибка: ровно одна константа USE_UDP / USE_UART должна быть True",
              file=sys.stderr)
        sys.exit(1)
    return mavutil.mavlink_connection(addr, force_connected=True)


def send_heading(conn, azimuth_deg: float,
                 x: float, y: float, z: float) -> None:
    """ATT_POS_MOCAP (#138) с явной ковариацией.

    ArduPilot параметры:
        EK3_SRC1_POSXY = 6  (ExternalNav)
        EK3_SRC1_POSZ  = 6
        EK3_SRC1_YAW   = 6
        VISO_TYPE      = 1  (MAVLink)
        VISO_DELAY_MS  = 50
    """
    yaw_rad = math.radians(azimuth_deg)
    q = [
        math.cos(yaw_rad / 2),  # w
        0.0,                     # x  (roll=0)
        0.0,                     # y  (pitch=0)
        math.sin(yaw_rad / 2),  # z
    ]
    usec = int(time.time() * 1e6)
    global _ATT_POS_MOCAP_SUPPORTS_COVARIANCE
    if _ATT_POS_MOCAP_SUPPORTS_COVARIANCE:
        try:
            conn.mav.att_pos_mocap_send(
                usec,
                q,
                x, y, z,
                covariance=COVARIANCE,
            )
            return
        except TypeError:
            _ATT_POS_MOCAP_SUPPORTS_COVARIANCE = False
            print("  [WARN] pymavlink без covariance для ATT_POS_MOCAP, отправляю без нее")

    conn.mav.att_pos_mocap_send(
        usec,
        q,
        x, y, z,
    )


def send_override(conn, pwm6: int, pwm7: int) -> None:
    conn.mav.rc_channels_override_send(
        1, 1,
        0, 0, 0, 0, 0,
        pwm6, pwm7, 0,
    )


def send_gps_origin(conn, lat_deg: float, lon_deg: float, alt_m: float = 0.0) -> None:
    conn.mav.set_gps_global_origin_send(
        1,
        int(lat_deg * 1e7),
        int(lon_deg * 1e7),
        int(alt_m * 1000),
    )


def pose_to_ardupilot(x_m: float, y_m: float, z_m: float):
    """ArUco (X right, Y forward, Z up) → ArduPilot NED (X fwd, Y right, Z down)."""
    return y_m, x_m, -z_m


# ╔══════════════════════════════════════════════════════════════════════════════
# ║  Главный цикл
# ╚══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    conn = make_connection()
    print(f"RC_OVERRIDE  CH6={PWM_CH6}  CH7={PWM_CH7}  AZIMUTH_BACKUP={AZIMUTH_DEG}°")
    print("Ctrl+C — остановить")

    send_gps_origin(conn, lat_deg=60.00, lon_deg=30.0, alt_m=150.0)
    print("GPS origin установлен")

    tracker     = None
    pose_worker = None

    try:
        tracker = ArucoPoseTracker()
        print(f"ArUco трекер запущен  |  {tracker.get_capture_info()}")

        pose_worker = PoseWorker(tracker)
        pose_worker.start()

        ema              = EMAFilter()
        sent             = 0
        last_rate_time   = time.perf_counter()
        last_vision_frames = 0

        # Состояние для защиты от выбросов
        prev_raw_xy      = None   # (x, y) предыдущего принятого измерения
        prev_yaw_deg     = None
        reject_count     = 0

        # Состояние last_pose с временной меткой
        last_pose        = None
        last_pose_time   = -999.0
        frozen_yaw_deg   = float(AZIMUTH_DEG)

        while True:
            t0 = time.perf_counter()

            pose, marker_visible, vision_frames, vision_error = pose_worker.snapshot()
            if vision_error is not None:
                raise vision_error

            # ── Принять новое измерение (с проверкой выброса) ─────────────────
            if pose is not None:
                x_m, y_m, z_m, yaw_raw = pose

                # Если поза долго не обновлялась, новое измерение считаем новой опорной точкой.
                stale_pose = (time.perf_counter() - last_pose_time) >= POSE_TIMEOUT_SEC
                if stale_pose:
                    ema.reset()
                    prev_raw_xy = None
                    prev_yaw_deg = None
                    reject_count = 0

                # Проверка прыжка позиции
                pos_ok = True
                if prev_raw_xy is not None:
                    jump = math.hypot(x_m - prev_raw_xy[0], y_m - prev_raw_xy[1])
                    if jump > MAX_POS_JUMP_M:
                        pos_ok = False
                        print(f"  [SKIP] pos jump {jump*100:.1f} cm > {MAX_POS_JUMP_M*100:.0f} cm")

                # Проверка прыжка yaw
                yaw_adjusted = (yaw_raw + CAMERA_AZIMUTH_OFFSET_DEG) % 360.0
                yaw_ok = True
                if prev_yaw_deg is not None:
                    yaw_jump = abs(_angle_diff_deg(yaw_adjusted, prev_yaw_deg))
                    if yaw_jump > MAX_YAW_JUMP_DEG:
                        yaw_ok = False
                        print(f"  [SKIP] yaw jump {yaw_jump:.1f}° > {MAX_YAW_JUMP_DEG:.0f}°")

                if pos_ok and yaw_ok:
                    prev_raw_xy  = (x_m, y_m)
                    prev_yaw_deg = yaw_adjusted
                    ema.update(x_m, y_m, z_m, yaw_adjusted)
                    last_pose      = ema.get()
                    last_pose_time = time.perf_counter()
                    frozen_yaw_deg = last_pose[3]
                    reject_count   = 0
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

            # ── Определить текущую позу для отправки ──────────────────────────
            pose_age = time.perf_counter() - last_pose_time

            if last_pose is not None and pose_age < POSE_TIMEOUT_SEC:
                x_f, y_f, z_f, yaw_f = last_pose
                if marker_visible:
                    src = "aruco"
                else:
                    src = f"last({pose_age:.1f}s)"
            else:
                # Нет свежей позы — позицию не доверяем, yaw замораживаем на последнем принятом.
                x_f, y_f, z_f = 0.0, 0.0, 0.0
                yaw_f = frozen_yaw_deg
                src   = "резерв"

            ap_x, ap_y, ap_z = pose_to_ardupilot(x_f, y_f, z_f)

            send_heading(conn, yaw_f, ap_x, ap_y, ap_z)
            sent += 1

            # ── Статистика каждые 50 пакетов (≈ 1 сек) ───────────────────────
            if sent % 50 == 0:
                now = time.perf_counter()
                dt  = now - last_rate_time
                real_hz   = 50.0 / dt if dt > 0 else 0.0
                vision_hz = (vision_frames - last_vision_frames) / dt if dt > 0 else 0.0
                last_rate_time     = now
                last_vision_frames = vision_frames

                hi = lo = ""
                extra = ""
                if prev_yaw_deg is not None and ema.initialised:
                    _, _, _, yaw_ema = ema.get()
                    diff = abs(_angle_diff_deg(yaw_f, yaw_ema))
                    if diff > 3.0:
                        hi, lo = "\033[93m", "\033[0m"
                        extra = f"  ⚠ yaw_diff={diff:.1f}°"

                print(
                    f"{hi}  sent={sent:5d}"
                    f"  hz={real_hz:.1f}/{vision_hz:.1f}"
                    f"  [{src}]"
                    f"  x={x_f:.3f}m y={y_f:.3f}m z={z_f:.3f}m"
                    f"  ap_x={ap_x:.3f} ap_y={ap_y:.3f} ap_z={ap_z:.3f}"
                    f"  az={yaw_f:.1f}°"
                    f"{extra}{lo}"
                )

            # ── Выдержать период ──────────────────────────────────────────────
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
