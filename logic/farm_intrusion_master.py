import time
import spidev
import lgpio
import board
import busio
import adafruit_mlx90640
import subprocess
import os
import csv
import json
import shutil
import numpy as np
import tensorflow as tf
from datetime import datetime
from ultralytics import YOLO

BASE_DIR              = "/home/aivisionhub/wildlife_project"
ACTIVE_YOLO_PATH      = f"{BASE_DIR}/yolov8n_active.pt"
BASE_YOLO_PATH        = f"{BASE_DIR}/yolov8n_base.pt"
MOBILENET_PATH        = f"{BASE_DIR}/model2_mobilenet.keras"
MOBILENET_LABELS      = f"{BASE_DIR}/class_labels.txt"
LOG_FILE              = f"{BASE_DIR}/activity_log.csv"
BASELINE_LOG_FILE     = f"{BASE_DIR}/thermal_baselines.csv"
INTRUSION_COUNT_FILE  = f"{BASE_DIR}/intrusion_count.json"

os.makedirs(f"{BASE_DIR}/captures",       exist_ok=True)
os.makedirs(f"{BASE_DIR}/thermal_frames", exist_ok=True)
os.makedirs(f"{BASE_DIR}/tmp",            exist_ok=True)

def _initial_yolo_path():
    if os.path.exists(ACTIVE_YOLO_PATH):
        return ACTIVE_YOLO_PATH
    if os.path.exists(BASE_YOLO_PATH):
        return BASE_YOLO_PATH
    return "yolov8n.pt"

_startup_yolo = _initial_yolo_path()
yolo_model    = YOLO(_startup_yolo)
print(f"[YOLO] Loaded: {_startup_yolo}")

mobilenet_model  = None
mobilenet_labels = []

if os.path.exists(MOBILENET_PATH):
    mobilenet_model = tf.keras.models.load_model(MOBILENET_PATH)
    print(f"[MOBILENET] Loaded: {MOBILENET_PATH}")
    if os.path.exists(MOBILENET_LABELS):
        with open(MOBILENET_LABELS) as f:
            mobilenet_labels = [l.strip() for l in f.readlines()]
        print(f"[MOBILENET] Classes: {mobilenet_labels}")
else:
    print(f"[MOBILENET] Not found at {MOBILENET_PATH} — using YOLO only")

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="") as f:
        csv.writer(f).writerow([
            "Date", "Entry_Time", "Exit_Time", "Duration_s",
            "YOLO_Label", "YOLO_Conf", "MobileNet_Label", "MobileNet_Conf",
            "Final_Label", "Scene_Mean_C", "Hot_Pixel_Count",
            "Max_Temp_C", "Trigger_Type", "Deterrent", "Status"
        ])

if not os.path.exists(BASELINE_LOG_FILE):
    with open(BASELINE_LOG_FILE, "w", newline="") as f:
        csv.writer(f).writerow([
            "Timestamp", "Baseline_Mean_C", "Baseline_Std_C",
            "Baseline_Min_C", "Baseline_Max_C", "Frames_Used"
        ])

BODY_TEMP_MIN       = 30.0
MIN_HOT_PIXELS      = 20
THERMAL_HZ          = 2
CONFIDENCE_YOLO     = 0.20
CONFIDENCE_MOBILENET= 0.55
CAM_WIDTH           = 640
CAM_HEIGHT          = 480
MISS_THRESHOLD      = 3
MAX_SESSION_S       = 1800
TRACKING_INTERVAL   = 1.0
IDLE_INTERVAL       = 0.5
DEBUG_EVERY         = 10
BASELINE_INTERVAL_S = 3600
BASELINE_FRAMES     = 30
BASELINE_FPS        = 2
SAVE_THERMAL_FRAMES = True
RETRAIN_TRIGGER     = 50

DAY_START_HOUR      = 6    # 06:00 AM — daytime begins
DAY_END_HOUR        = 19   # 07:00 PM — daytime ends

HUMAN_CLASSES = {
    "Person", "Man", "Woman", "Boy", "Girl",
    "Human body", "Human face", "Human head", "human",
}

def is_daytime():
    h = datetime.now().hour
    return DAY_START_HOUR <= h < DAY_END_HOUR

def get_deterrent_mode(label):
    mode = DETERRENT_MAP.get(label)
    if label in HUMAN_CLASSES:
        mode = None if is_daytime() else "lights"
    if is_daytime():
        if mode == "lights":
            return None       # lights-only animals: log only during day
        if mode == "both":
            return "siren"    # both-mode animals: siren only during day
    return mode

DETERRENT_MAP = {
    "Person": "both", "Man": "both", "Woman": "both",
    "Boy": "both", "Girl": "both", "Human body": "both",
    "Human face": "both", "Human head": "both",
    "Elephant": "both", "Bear": "both", "Brown bear": "both",
    "Tiger": "both", "Lion": "both", "Leopard": "both",
    "Jaguar (Animal)": "both", "Cheetah": "both",
    "Rhinoceros": "both", "Hippopotamus": "both",
    "Pig": "both", "Bull": "both", "Cattle": "both",
    "Camel": "both", "Dog": "both", "Crocodile": "both",
    "Car": "both", "Truck": "both", "Van": "both",
    "Motorcycle": "both", "Bus": "both", "Taxi": "both",
    "Goat": "siren", "Deer": "siren", "Antelope": "siren",
    "Horse": "siren", "Mule": "siren", "Sheep": "siren",
    "Kangaroo": "siren", "Alpaca": "siren", "Porcupine": "siren",
    "Lynx": "siren", "Ostrich": "siren", "Zebra": "siren",
    "Monkey": "lights", "Cat": "lights", "Fox": "lights",
    "Rabbit": "lights", "Squirrel": "lights", "Raccoon": "lights",
    "Hedgehog": "lights", "Panda": "lights", "Red panda": "lights",
    "Otter": "lights", "Koala": "lights", "Skunk": "lights",
    "Bird": "lights", "Chicken": "lights", "Duck": "lights",
    "Goose": "lights", "Parrot": "lights", "Owl": "lights",
    "Eagle": "lights", "Falcon": "lights", "Sparrow": "lights",
    "Magpie": "lights", "Raven": "lights", "Turkey": "lights",
    "Swan": "lights", "Snake": "lights", "Lizard": "lights",
    "Tortoise": "lights", "Turtle": "lights",
    "Bee": "lights", "Ant": "lights", "Scorpion": "lights",
    "human": "both", "cow": "both", "bear": "both",
    "deer": "siren", "goat": "siren", "wild boar": "both",
}

WATCHED_ANIMALS = set(DETERRENT_MAP.keys())
SIREN_ANIMALS   = {k for k, v in DETERRENT_MAP.items() if v in ("siren", "both")}
LIGHTS_ANIMALS  = {k for k, v in DETERRENT_MAP.items() if v in ("lights", "both")}

i2c = busio.I2C(board.SCL, board.SDA, frequency=400000)
mlx = adafruit_mlx90640.MLX90640(i2c)
_rate_map = {
    1: adafruit_mlx90640.RefreshRate.REFRESH_1_HZ,
    2: adafruit_mlx90640.RefreshRate.REFRESH_2_HZ,
    4: adafruit_mlx90640.RefreshRate.REFRESH_4_HZ,
    8: adafruit_mlx90640.RefreshRate.REFRESH_8_HZ,
}
mlx.refresh_rate = _rate_map.get(THERMAL_HZ, adafruit_mlx90640.RefreshRate.REFRESH_2_HZ)

def _open_gpio():
    RST = 23
    def _try_claim(h):
        lgpio.gpio_claim_output(h, RST)
    def _kill_daemons():
        os.system("sudo pkill -f lgpiod  2>/dev/null")
        os.system("sudo pkill -f pigpiod 2>/dev/null")
        time.sleep(0.4)
    def _kernel_release():
        ret = os.system("raspi-gpio set 23 ip 2>/dev/null")
        if ret != 0:
            os.system("pinctrl set 23 ip 2>/dev/null")
        time.sleep(0.15)

    h = lgpio.gpiochip_open(0)
    try:
        _try_claim(h); return h
    except lgpio.error:
        pass
    try: lgpio.gpiochip_close(h)
    except Exception: pass
    _kill_daemons()
    h = lgpio.gpiochip_open(0)
    try:
        _try_claim(h); return h
    except lgpio.error:
        pass
    try: lgpio.gpiochip_close(h)
    except Exception: pass
    _kernel_release()
    h = lgpio.gpiochip_open(0)
    try:
        _try_claim(h); return h
    except lgpio.error as e:
        lgpio.gpiochip_close(h)
        raise RuntimeError(f"Cannot claim GPIO 23: {e}") from e

h   = _open_gpio()
spi = spidev.SpiDev()
spi.open(0, 0)
spi.max_speed_hz = 500000

def _wr(reg, val): spi.xfer2([reg | 0x80, val])
def _rd(reg):      return spi.xfer2([reg & 0x7F, 0x00])[1]

def init_lora():
    lgpio.gpio_write(h, 23, 0); time.sleep(0.01)
    lgpio.gpio_write(h, 23, 1); time.sleep(0.1)
    _wr(0x01, 0x80); time.sleep(0.01)
    _wr(0x06, 0x6C); _wr(0x07, 0x40); _wr(0x08, 0x00)
    _wr(0x09, 0xFF); _wr(0x1E, 0x74); _wr(0x1D, 0x72)
    _wr(0x39, 0x12)
    _wr(0x01, 0x81)
    print("LoRa ready.")

def _tx_done(timeout=2.0):
    t = time.time() + timeout
    while time.time() < t:
        if _rd(0x12) & 0x08:
            _wr(0x12, 0xFF); return True
        time.sleep(0.01)
    _wr(0x12, 0xFF); return False

def send_cmd(cmd):
    p = list(cmd.encode())
    _wr(0x01, 0x81); _wr(0x0D, 0x00); _wr(0x0E, 0x00)
    _wr(0x22, len(p)); spi.xfer2([0x80] + p)
    _wr(0x01, 0x83); _tx_done()

def send_heartbeat(): send_cmd("HEARTBEAT")

def capture(path):
    try:
        subprocess.run([
            "rpicam-still", "-o", path, "--nopreview",
            "--immediate", "-t", "1",
            "--width", str(CAM_WIDTH), "--height", str(CAM_HEIGHT),
            "--awb", "auto", "--denoise", "off",
        ], check=True, timeout=10)
        return True
    except Exception as e:
        print(f"[WARN] Camera: {e}"); return False

def detect_yolo(path):
    try:
        results    = yolo_model(path, conf=CONFIDENCE_YOLO, save=False, verbose=False)
        best, conf = None, 0.0
        for r in results:
            for box in r.boxes:
                c = float(box.conf[0])
                if c > conf:
                    conf = c
                    best = yolo_model.names[int(box.cls[0])]
        return best, conf
    except Exception as e:
        print(f"[WARN] YOLO: {e}"); return None, 0.0

def detect_mobilenet(path):
    if mobilenet_model is None:
        return None, 0.0
    try:
        import cv2
        img    = cv2.imread(path)
        img    = cv2.resize(img, (224, 224))
        img    = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        arr    = img.astype(np.float32) / 255.0
        proba  = mobilenet_model.predict(arr[np.newaxis], verbose=0)[0]
        idx    = int(np.argmax(proba))
        conf   = float(proba[idx])
        label  = mobilenet_labels[idx] if idx < len(mobilenet_labels) else str(idx)
        if conf < CONFIDENCE_MOBILENET:
            return None, conf
        return label, conf
    except Exception as e:
        print(f"[WARN] MobileNet: {e}"); return None, 0.0

def fuse_predictions(yolo_label, yolo_conf, mn_label, mn_conf):
    if yolo_label and mn_label:
        if yolo_label.lower() == mn_label.lower():
            return yolo_label, max(yolo_conf, mn_conf)
        if mn_conf > yolo_conf + 0.15:
            return mn_label, mn_conf
        return yolo_label, yolo_conf
    if yolo_label:
        return yolo_label, yolo_conf
    if mn_label:
        return mn_label, mn_conf
    return None, 0.0

def save_detection_image(tmp_path, label, date_str):
    save_dir = f"{BASE_DIR}/captures/{date_str}"
    os.makedirs(save_dir, exist_ok=True)
    ts  = datetime.now().strftime("%H%M%S_%f")[:9]
    dst = f"{save_dir}/{ts}_{label}.jpg"
    try:
        shutil.copy2(tmp_path, dst)
    except Exception as e:
        print(f"[WARN] Image save: {e}")

_baseline_mean      = None
_baseline_std       = None
_last_baseline_time = 0.0

def _raw_frame():
    buf = [0.0] * 768
    try:
        mlx.getFrame(buf)
        return np.array(buf, dtype=np.float32)
    except Exception as e:
        print(f"[WARN] Thermal: {e}"); return None

def run_baseline_calibration():
    global _baseline_mean, _baseline_std, _last_baseline_time
    ts     = datetime.now()
    frames = []
    for _ in range(BASELINE_FRAMES):
        arr = _raw_frame()
        if arr is not None:
            frames.append(arr)
        time.sleep(1.0 / BASELINE_FPS)
    if len(frames) < 5:
        return
    stack          = np.stack(frames)
    mean_frame     = stack.mean(axis=0)
    _baseline_mean = float(mean_frame.mean())
    _baseline_std  = float(mean_frame.std())
    _last_baseline_time = time.time()
    with open(BASELINE_LOG_FILE, "a", newline="") as f:
        csv.writer(f).writerow([
            ts.strftime("%Y-%m-%d %H:%M:%S"),
            f"{_baseline_mean:.2f}", f"{_baseline_std:.2f}",
            f"{mean_frame.min():.2f}", f"{mean_frame.max():.2f}",
            len(frames)
        ])
    print(f"[BASELINE] mean={_baseline_mean:.1f}C  std={_baseline_std:.1f}C")

def read_thermal():
    arr = _raw_frame()
    if arr is None:
        return None
    scene_mean      = float(arr.mean())
    hot_pixel_count = int(np.sum(arr >= BODY_TEMP_MIN))
    max_temp        = float(arr.max())
    triggered       = hot_pixel_count >= MIN_HOT_PIXELS
    return arr, scene_mean, hot_pixel_count, max_temp, triggered

def save_thermal_frame(arr, ts_str, label):
    if not SAVE_THERMAL_FRAMES:
        return
    np.save(f"{BASE_DIR}/thermal_frames/{ts_str}_{label}.npy", arr.reshape(24, 32))

def activate_deterrent(mode):
    if mode == "siren":
        send_cmd("ACTIVATE_SIREN")
    elif mode == "lights":
        send_cmd("ACTIVATE_LIGHTS")
    elif mode == "both":
        send_cmd("ACTIVATE_BOTH")
    return mode

def deactivate_deterrent(mode):
    if mode == "siren":
        send_cmd("DEACTIVATE_SIREN")
    elif mode == "lights":
        send_cmd("DEACTIVATE_LIGHTS")
    elif mode == "both":
        send_cmd("DEACTIVATE_BOTH")

def _load_intrusion_count():
    if os.path.exists(INTRUSION_COUNT_FILE):
        with open(INTRUSION_COUNT_FILE) as f:
            return json.load(f).get("count", 0)
    return 0

def _save_intrusion_count(count):
    with open(INTRUSION_COUNT_FILE, "w") as f:
        json.dump({"count": count, "updated": datetime.now().isoformat()}, f)

def _trigger_retrain_if_needed():
    count = _load_intrusion_count() + 1
    _save_intrusion_count(count)
    print(f"[RETRAIN] Intrusion count: {count}/{RETRAIN_TRIGGER}")
    if count >= RETRAIN_TRIGGER:
        print(f"[RETRAIN] {RETRAIN_TRIGGER} intrusions reached — triggering adaptive retrain now...")
        retrain_script = f"{BASE_DIR}/adaptive_retrain.py"
        if os.path.exists(retrain_script):
            subprocess.Popen(
                ["python3", retrain_script, "--force"],
                stdout=open(f"{BASE_DIR}/retrain_triggered.log", "a"),
                stderr=subprocess.STDOUT
            )
        _save_intrusion_count(0)
        print("[RETRAIN] Retrain process launched in background. Counter reset.")

def run_session(start_time, trigger_arr, scene_mean, hot_px, max_temp, trigger_type):
    entry_ts     = start_time.strftime("%H:%M:%S")
    date_str     = start_time.strftime("%Y-%m-%d")
    session_id   = start_time.strftime("%Y%m%d_%H%M%S")
    final_label  = "Unknown"
    best_conf    = 0.0
    yolo_label   = None; yolo_conf   = 0.0
    mn_label     = None; mn_conf     = 0.0
    misses       = 0
    active_mode  = None
    deterrent_on = False
    saved_count  = 0
    intrusion_logged = False

    print(f"\n[SESSION] {session_id} | scene_avg={scene_mean:.1f}C  hot_px={hot_px}")

    path = f"{BASE_DIR}/tmp/{session_id}_s0.jpg"
    if capture(path):
        yolo_label, yolo_conf = detect_yolo(path)
        mn_label,   mn_conf   = detect_mobilenet(path)
        fused, fused_conf     = fuse_predictions(yolo_label, yolo_conf, mn_label, mn_conf)

        print(f"[SESSION] YOLO={yolo_label}({yolo_conf:.0%})  "
              f"MobileNet={mn_label}({mn_conf:.0%})  "
              f"Fused={fused}({fused_conf:.0%})")

        mode = get_deterrent_mode(fused) if fused else None
        if fused and (fused in DETERRENT_MAP or fused in HUMAN_CLASSES) and mode is not None:
            final_label  = fused
            best_conf    = fused_conf
            active_mode  = mode
            deterrent_on = True
            misses       = 0
            activate_deterrent(active_mode)
        elif fused and fused in HUMAN_CLASSES and is_daytime():
            final_label = fused
            best_conf   = fused_conf
            misses      = 0
            print(f"[SESSION] HUMAN detected in daytime — logging only, no deterrent")
            save_detection_image(path, final_label, date_str)
            save_thermal_frame(trigger_arr, session_id, f"trigger_{final_label}")
            saved_count += 1
            if not intrusion_logged:
                _trigger_retrain_if_needed()
                intrusion_logged = True
        else:
            misses = 1
        if os.path.exists(path): os.remove(path)

    frame_idx = 1
    while True:
        if (datetime.now() - start_time).total_seconds() > MAX_SESSION_S:
            break

        time.sleep(TRACKING_INTERVAL)
        path = f"{BASE_DIR}/tmp/{session_id}_s{frame_idx}.jpg"
        frame_idx += 1

        if not capture(path):
            misses += 1
            if os.path.exists(path): os.remove(path)
            continue

        yolo_label, yolo_conf = detect_yolo(path)
        mn_label,   mn_conf   = detect_mobilenet(path)
        fused, fused_conf     = fuse_predictions(yolo_label, yolo_conf, mn_label, mn_conf)

        mode = get_deterrent_mode(fused) if fused else None
        if fused and (fused in DETERRENT_MAP or fused in HUMAN_CLASSES):
            if mode is not None:
                activate_deterrent(mode)
                active_mode  = mode
                deterrent_on = True
            elif fused in HUMAN_CLASSES and is_daytime():
                print(f"[SESSION] HUMAN in daytime — log only")
            save_detection_image(path, fused, date_str)
            arr = _raw_frame()
            if arr is not None:
                save_thermal_frame(arr, f"{session_id}_f{frame_idx}", fused)
            final_label  = fused
            best_conf    = max(best_conf, fused_conf)
            misses       = 0
            saved_count += 1
        else:
            misses += 1

        if os.path.exists(path): os.remove(path)

        if misses >= MISS_THRESHOLD:
            break

    if deterrent_on and active_mode:
        deactivate_deterrent(active_mode)

    exit_time = datetime.now()
    duration  = (exit_time - start_time).total_seconds()

    with open(LOG_FILE, "a", newline="") as f:
        csv.writer(f).writerow([
            date_str, entry_ts, exit_time.strftime("%H:%M:%S"),
            f"{duration:.1f}", yolo_label, f"{yolo_conf:.0%}",
            mn_label, f"{mn_conf:.0%}", final_label,
            f"{scene_mean:.1f}", hot_px, f"{max_temp:.1f}",
            trigger_type, active_mode or "none", "Cleared"
        ])

    print(f"[SESSION] Done. label={final_label} duration={duration:.1f}s saved={saved_count}\n")

_model_mtime = os.path.getmtime(_startup_yolo) if os.path.exists(_startup_yolo) else 0.0

def _check_model_update():
    global yolo_model, _model_mtime
    if not os.path.exists(ACTIVE_YOLO_PATH):
        return
    try:
        mtime = os.path.getmtime(ACTIVE_YOLO_PATH)
    except OSError:
        return
    if mtime <= _model_mtime:
        return
    time.sleep(2)
    try:
        yolo_model   = YOLO(ACTIVE_YOLO_PATH)
        _model_mtime = mtime
        print(f"[YOLO] Updated model loaded: {ACTIVE_YOLO_PATH}")
    except Exception as e:
        print(f"[YOLO] Reload failed: {e}")

def main():
    init_lora()
    run_baseline_calibration()

    last_hb          = time.time()
    last_model_check = time.time()
    idle_count       = 0

    print("=" * 60)
    print(" Farm Intrusion System — ONLINE")
    print(f" Models  : YOLOv8n + MobileNetV2 (fused)")
    print(f" Thermal : >={MIN_HOT_PIXELS} pixels above {BODY_TEMP_MIN}C")
    print(f" Retrain : every {RETRAIN_TRIGGER} confirmed intrusions")
    print("=" * 60 + "\n")

    try:
        while True:
            if time.time() - _last_baseline_time >= BASELINE_INTERVAL_S:
                run_baseline_calibration()

            if time.time() - last_model_check >= 300:
                _check_model_update()
                last_model_check = time.time()

            result = read_thermal()
            if result is not None:
                arr, scene_mean, hot_px, max_temp, triggered = result
                idle_count += 1

                if DEBUG_EVERY and (idle_count % DEBUG_EVERY == 0):
                    print(f"[THERMAL] avg={scene_mean:.1f}C  max={max_temp:.1f}C  "
                          f"hot_px={hot_px}  -> {'TRIGGER' if triggered else 'idle'}")

                if triggered:
                    run_session(datetime.now(), arr, scene_mean, hot_px, max_temp, "body_temp")
                    idle_count = 0

            now = time.time()
            if now - last_hb >= 30:
                send_heartbeat()
                last_hb = now

            time.sleep(IDLE_INTERVAL)

    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        try:
            lgpio.gpio_free(h, 23)
            lgpio.gpiochip_close(h)
        except Exception: pass
        try: spi.close()
        except Exception: pass
        print("[CLEANUP] Done.")

if __name__ == "__main__":
    main()
