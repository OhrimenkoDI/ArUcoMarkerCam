import sys
import time

from pymavlink import mavutil

# ── Выбор интерфейса ──────────────────────────────────────────────────────────
# Установи одну из констант в True, вторую в False

USE_UDP  = False    # UDP: подключение через MAVProxy (--out=udpin:0.0.0.0:14552)
USE_UART = True   # UART: прямое подключение к полётному контроллеру по COM-порту

# ── Настройки UDP ─────────────────────────────────────────────────────────────
UDP_IP   = "10.0.20.15"
UDP_PORT = 14550

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


def send_override(conn, pwm6: int, pwm7: int) -> None:
    conn.mav.rc_channels_override_send(
        1, 1,             # target_system, target_component
        0, 0, 0, 0, 0,   # CH1-CH5: 0 = не переопределять
        pwm6,             # CH6
        pwm7,             # CH7
        0,                # CH8
    )


def main() -> None:
    conn = make_connection()
    print(f"RC_OVERRIDE  CH6={PWM_CH6}  CH7={PWM_CH7}")
    print("Ctrl+C — остановить и отпустить каналы")

    sent = 0
    try:
        while True:
            send_override(conn, PWM_CH6, PWM_CH7)
            sent += 1
            if sent % 50 == 0:
                print(f"  sent={sent}  CH6={PWM_CH6}  CH7={PWM_CH7}")
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
