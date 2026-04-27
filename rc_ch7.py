import math
import sys
import time

from pymavlink import mavutil

# ── Выбор интерфейса ──────────────────────────────────────────────────────────
# Установи одну из констант в True, вторую в False

USE_UDP  = True    # UDP: подключение через MAVProxy (--out=udpin:0.0.0.0:14552)
USE_UART = False   # UART: прямое подключение к полётному контроллеру по COM-порту

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
AZIMUTH_DEG = 10  # градусы, потом сделаю переменным
# ─────────────────────────────────────────────────────────────────────────────


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
    print(f"RC_OVERRIDE  CH6={PWM_CH6}  CH7={PWM_CH7}  AZIMUTH={AZIMUTH_DEG}°")
    print("Ctrl+C — остановить и отпустить каналы")


    # Задаём origin — реальные координаты твоей точки старта
    send_gps_origin(conn, lat_deg=60.00, lon_deg=30.0, alt_m=150.0)
    print("GPS origin установлен")


    sent = 0
    try:
        while True:
            #send_override(conn, PWM_CH6, PWM_CH7)
            send_heading(conn, AZIMUTH_DEG)
            sent += 1
            if sent % 50 == 0:
                print(f"  sent={sent}  CH6={PWM_CH6}  CH7={PWM_CH7}  az={AZIMUTH_DEG}°")
            time.sleep(INTERVAL)

    except KeyboardInterrupt:
        print("\nОтпускаю каналы...")
        for _ in range(5):
            send_override(conn, 0, 0)
            time.sleep(INTERVAL)

    finally:
        conn.close()
        print("Закрыто.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
