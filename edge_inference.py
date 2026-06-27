"""
edge_inference.py
═══════════════════════════════════════════════════════════════════
EDGE DEPLOYMENT — Raspberry Pi 5 Node Controller
Farm Animal Intrusion Prevention System

Hardware pipeline:
  1. Thermal camera detects movement → GPIO trigger wakes this script
  2. Raspberry Pi Camera Module v2 (Night-vision) captures a frame
  3. TFLite model runs inference on the captured frame
  4. Decision logic:
       HUMAN detected  → day  : log only (no alert)
                       → night: ACTIVATE LIGHT (illuminate, do not alarm)
       Dangerous animal (conf ≥ 0.75) → ACTIVATE BOTH
       Dangerous animal (conf <  0.75) → ACTIVATE SIREN
       Non-dangerous animal            → ACTIVATE LIGHT
  5. Command sent wirelessly to ESP32 over LoRa (no internet required)
  6. Heartbeat "ALIVE" sent to ESP32 via LoRa every 2 minutes.
     ESP32 watchdog shuts actuators if ALIVE stops for > 2 min.

LoRa:
  Pi 5 ← SX127x LoRa HAT (SPI bus)
  ESP32 ← LoRa module (SPI)
  Frequency: 433 MHz (or 868 / 915 MHz depending on region)

Usage:
  python edge_inference.py --model mobilenet_farm.tflite
  python edge_inference.py --model mobilenet_farm.tflite --simulate
  python edge_inference.py --model mobilenet_farm.tflite --demo --simulate
═══════════════════════════════════════════════════════════════════
"""

import cv2
import numpy as np
import argparse
import time
import json
import threading
from datetime import datetime

# ── GPIO (Raspberry Pi 5) ────────────────────────────────────────
try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True

    THERMAL_TRIGGER_PIN = 17   # INPUT  — thermal camera motion signal
    STATUS_LED_PIN      = 27   # OUTPUT — heartbeat / status LED

    GPIO.setmode(GPIO.BCM)
    GPIO.setup(THERMAL_TRIGGER_PIN, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
    GPIO.setup(STATUS_LED_PIN, GPIO.OUT, initial=GPIO.LOW)

except ImportError:
    GPIO_AVAILABLE = False
    print("[INFO] RPi.GPIO not available — GPIO simulation mode")

# ── PiCamera Module v2 ───────────────────────────────────────────
try:
    from picamera2 import Picamera2
    PICAMERA_AVAILABLE = True
except ImportError:
    PICAMERA_AVAILABLE = False
    print("[INFO] picamera2 not available — OpenCV fallback")

# ── LoRa (SX127x via pyLoRa / SPI) ──────────────────────────────
try:
    from SX127x.LoRa import LoRa as _LoRaBase
    from SX127x.board_config import BOARD as _BOARD
    _BOARD.setup()
    LORA_AVAILABLE = True
except Exception:
    LORA_AVAILABLE = False
    print("[INFO] LoRa (pyLoRa/SX127x) not available — simulation mode")


# ─────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────
ANIMAL_CLASSES = [
    # Human intruder
    "human",
    # Farm / Livestock
    "cow", "goat", "sheep", "pig", "horse",
    "donkey", "chicken", "duck", "turkey",
    "rabbit", "alpaca", "llama",
    # Common Wild Predators
    "dog", "cat", "fox", "wolf", "coyote",
    "jackal", "hyena",
    # Large Wild Animals
    "bear", "deer", "wild_boar", "elephant",
    "bison", "moose", "elk", "antelope",
    "kangaroo", "zebra", "rhinoceros",
    # Big Cats
    "lion", "tiger", "leopard", "cheetah",
    "puma", "lynx",
    # Other Wildlife
    "monkey", "baboon", "raccoon",
    "snake", "eagle", "vulture",
]

# NOTE: "human" is handled separately — never triggers siren/both
DANGEROUS_ANIMALS = {
    "dog", "fox", "wolf", "coyote", "jackal", "hyena",
    "lion", "tiger", "leopard", "cheetah", "puma", "lynx",
    "bear", "wild_boar", "elephant", "rhinoceros",
    "monkey", "baboon", "snake",
}

CONFIDENCE_THRESHOLD = 0.55
ALERT_COOLDOWN_SEC   = 10
HEARTBEAT_INTERVAL   = 120      # seconds — must match ESP32 watchdog timeout
IMG_SIZE             = (224, 224)

# Night hours: 20:00 → 06:00
NIGHT_START_HOUR = 20
NIGHT_END_HOUR   = 6


# ─────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────
def is_nighttime() -> bool:
    """Returns True if current time is within night hours."""
    h = datetime.now().hour
    return h >= NIGHT_START_HOUR or h < NIGHT_END_HOUR


# ─────────────────────────────────────────────────────────────────
# LORA COMMUNICATOR
# ─────────────────────────────────────────────────────────────────
class _LoRaSender(_LoRaBase if LORA_AVAILABLE else object):
    """
    Thin LoRa transmitter wrapper around pyLoRa / SX127x.
    Configured for transmit-only (one-way Pi → ESP32).
    """

    def __init__(self, verbose=False):
        if LORA_AVAILABLE:
            super().__init__(verbose=verbose)
            self.set_mode(0x01)               # STDBY
            self.set_freq(433.0)              # MHz — change to 868/915 for your region
            self.set_spreading_factor(7)
            self.set_bandwidth(125e3)
            self.set_coding_rate(5)
            self.set_preamble(8)
            self.set_sync_word(0x12)

    def transmit(self, message: str):
        if LORA_AVAILABLE:
            payload = [ord(c) for c in message]
            self.write_payload(payload)
            self.set_mode(0x03)               # TX
            time.sleep(0.2)
            self.set_mode(0x01)               # back to STDBY


class LoRaCommunicator:
    """
    Sends commands to the ESP32 actuator node over LoRa.
    No internet required — direct RF link.

    Commands:
        ACTIVATE LIGHT    → floodlight on
        ACTIVATE SIREN    → siren on
        ACTIVATE BOTH     → light + siren
        DEACTIVATE        → all off
        ALIVE             → heartbeat (every 2 min)
    """

    def __init__(self, simulate=False):
        self.simulate = simulate or not LORA_AVAILABLE
        self._lora    = None

        if not self.simulate:
            try:
                self._lora = _LoRaSender(verbose=False)
                print("[LoRa] Transmitter ready (SX127x)")
            except Exception as e:
                print(f"[LoRa] Init error: {e} — falling back to simulation")
                self.simulate = True

    def send(self, command: str):
        ts = datetime.now().strftime("%H:%M:%S")
        if self.simulate:
            print(f"  [LoRa SIM {ts}] → {command}")
        else:
            try:
                self._lora.transmit(command)
                print(f"  [LoRa SENT {ts}] → {command}")
            except Exception as e:
                print(f"  [LoRa ERROR] {e}")

    def activate(self, animal: str, confidence: float) -> str:
        """
        Decide and send the right activation signal.
        Human is NEVER passed here — it is handled before this call.
        """
        if animal in DANGEROUS_ANIMALS and confidence >= 0.75:
            self.send("ACTIVATE BOTH")
            return "ACTIVATE BOTH"
        elif animal in DANGEROUS_ANIMALS:
            self.send("ACTIVATE SIREN")
            return "ACTIVATE SIREN"
        else:
            self.send("ACTIVATE LIGHT")
            return "ACTIVATE LIGHT"

    def deactivate(self):
        self.send("DEACTIVATE")

    def alive(self):
        self.send("ALIVE")

    def close(self):
        if self._lora and LORA_AVAILABLE:
            try:
                _BOARD.teardown()
            except Exception:
                pass


# ─────────────────────────────────────────────────────────────────
# HEARTBEAT THREAD
# ─────────────────────────────────────────────────────────────────
class HeartbeatThread(threading.Thread):
    """
    Sends "ALIVE" to ESP32 via LoRa every HEARTBEAT_INTERVAL seconds.
    Runs as a daemon so it dies automatically when the main program exits.
    If this stops, the ESP32 watchdog shuts down all actuators after 2 min.
    """

    def __init__(self, lora: LoRaCommunicator, interval=HEARTBEAT_INTERVAL):
        super().__init__(daemon=True)
        self.lora     = lora
        self.interval = interval
        self._stop    = threading.Event()

    def run(self):
        print(f"[Heartbeat] Started — ALIVE every {self.interval}s via LoRa")
        while not self._stop.is_set():
            self.lora.alive()
            if GPIO_AVAILABLE:
                GPIO.output(STATUS_LED_PIN, GPIO.HIGH)
                time.sleep(0.1)
                GPIO.output(STATUS_LED_PIN, GPIO.LOW)
            self._stop.wait(self.interval)

    def stop(self):
        self._stop.set()


# ─────────────────────────────────────────────────────────────────
# CAMERA MANAGER (PiCamera v2 or OpenCV fallback)
# ─────────────────────────────────────────────────────────────────
class CameraManager:

    def __init__(self, cam_index=0, img_size=IMG_SIZE):
        self.img_size  = img_size
        self._picam    = None
        self._cap      = None
        self._use_pi   = PICAMERA_AVAILABLE

        if self._use_pi:
            self._picam = Picamera2()
            cfg = self._picam.create_still_configuration(
                main={"size": img_size, "format": "RGB888"}
            )
            self._picam.configure(cfg)
            print("[Camera] Raspberry Pi Camera Module v2 initialised")
        else:
            self._cap = cv2.VideoCapture(cam_index)
            print(f"[Camera] OpenCV fallback — camera {cam_index}")

    def start(self):
        if self._use_pi:
            self._picam.start()
            time.sleep(0.5)
        print("[Camera] Started")

    def capture(self) -> np.ndarray:
        """Returns an RGB uint8 image (H, W, 3)."""
        if self._use_pi:
            frame = self._picam.capture_array()
            return cv2.resize(frame, self.img_size)
        ret, frame = self._cap.read()
        if not ret:
            raise RuntimeError("Camera read failed")
        return cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), self.img_size)

    def stop(self):
        if self._use_pi and self._picam:
            self._picam.stop()
        if self._cap:
            self._cap.release()
        print("[Camera] Stopped")


# ─────────────────────────────────────────────────────────────────
# TFLITE INFERENCE ENGINE
# ─────────────────────────────────────────────────────────────────
class InferenceEngine:

    def __init__(self, model_path, class_names=ANIMAL_CLASSES,
                 conf_threshold=CONFIDENCE_THRESHOLD):
        self.class_names    = class_names
        self.conf_threshold = conf_threshold

        try:
            import tflite_runtime.interpreter as tflite
            self.interpreter = tflite.Interpreter(model_path=model_path)
        except ImportError:
            import tensorflow as tf
            self.interpreter = tf.lite.Interpreter(model_path=model_path)

        self.interpreter.allocate_tensors()
        self.input_det   = self.interpreter.get_input_details()[0]
        self.output_det  = self.interpreter.get_output_details()[0]
        self.input_shape = tuple(self.input_det['shape'][1:3])
        print(f"[Inference] Model loaded. Input: {self.input_shape}")

    def predict(self, rgb_image: np.ndarray):
        img = cv2.resize(rgb_image, (self.input_shape[1], self.input_shape[0]))
        img = img.astype(np.float32) / 255.0
        self.interpreter.set_tensor(self.input_det['index'], img[np.newaxis])
        self.interpreter.invoke()
        probs  = self.interpreter.get_tensor(self.output_det['index'])[0]
        pred   = int(np.argmax(probs))
        conf   = float(probs[pred])
        return self.class_names[pred], conf, probs


# ─────────────────────────────────────────────────────────────────
# MAIN NODE CONTROLLER
# ─────────────────────────────────────────────────────────────────
class NodeController:
    """
    Full pipeline:
      Thermal trigger (GPIO) → Camera capture → AI Inference
        → Decision logic → LoRa command to ESP32
        + background heartbeat thread
    """

    def __init__(self, model_path, cam_index=0,
                 conf_threshold=CONFIDENCE_THRESHOLD, simulate=False):
        self.conf_threshold = conf_threshold
        self.last_alert_t   = {}
        self.detection_log  = []

        self.lora    = LoRaCommunicator(simulate=simulate)
        self.camera  = CameraManager(cam_index)
        self.engine  = InferenceEngine(model_path,
                                       conf_threshold=conf_threshold)
        self.hb      = HeartbeatThread(self.lora)

    def start(self):
        self.camera.start()
        self.hb.start()
        self.lora.alive()   # immediate first heartbeat
        print("\n[Node] System ready. Waiting for thermal trigger …\n")

    # ── Single detection cycle ───────────────────────────────────
    def _on_trigger(self):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        night = is_nighttime()
        print(f"\n[{ts}] 🌡  Thermal trigger — {'night' if night else 'day'} mode")

        try:
            frame = self.camera.capture()
        except RuntimeError as e:
            print(f"[Camera] ERROR: {e}")
            return

        animal, conf, probs = self.engine.predict(frame)
        print(f"[Inference] {animal.upper()}  conf={conf*100:.1f}%")

        if conf >= self.conf_threshold:
            self._handle_detection(animal, conf, probs, ts, night)
        else:
            print(f"[Inference] Below threshold ({self.conf_threshold}) — ignored")

    # ── Decision logic ───────────────────────────────────────────
    def _handle_detection(self, animal, conf, probs, ts, night: bool):
        # Per-animal cooldown
        now = time.time()
        if now - self.last_alert_t.get(animal, 0) < ALERT_COOLDOWN_SEC:
            print(f"[Alert] Cooldown active for {animal} — skipped")
            return
        self.last_alert_t[animal] = now

        # ── HUMAN special case ───────────────────────────────────
        if animal == "human":
            if night:
                signal = "ACTIVATE LIGHT"
                self.lora.send(signal)
                action_desc = "Light ON (night deterrent)"
            else:
                signal      = "LOG ONLY"
                action_desc = "Log only (daytime human)"

            print(f"\n{'!'*55}")
            print(f"  👤 HUMAN DETECTED")
            print(f"  Action     : {action_desc}")
            print(f"  Time       : {ts} ({'NIGHT' if night else 'DAY'})")
            print(f"{'!'*55}\n")

            self._log(ts, animal, conf, signal, "👤 HUMAN")
            return

        # ── Animal signal logic ──────────────────────────────────
        priority = "🔴 HIGH" if animal in DANGEROUS_ANIMALS else "🟡 NORMAL"
        signal   = self.lora.activate(animal, conf)

        print(f"\n{'!'*55}")
        print(f"  🚨 INTRUSION ALERT [{priority}]")
        print(f"  Animal     : {animal.upper()}")
        print(f"  Confidence : {conf*100:.1f}%")
        print(f"  Signal     : {signal}")
        print(f"  Time       : {ts} ({'NIGHT' if night else 'DAY'})")
        print(f"{'!'*55}\n")

        top3 = sorted(enumerate(probs), key=lambda x: -x[1])[:3]
        for idx, p in top3:
            print(f"    {self.engine.class_names[idx]:12s}  {p*100:.1f}%")

        self._log(ts, animal, conf, signal, priority)

    def _log(self, ts, animal, conf, signal, priority):
        self.detection_log.append({
            "timestamp" : ts,
            "animal"    : animal,
            "confidence": round(conf, 4),
            "signal"    : signal,
            "priority"  : priority,
        })

    # ── Run modes ────────────────────────────────────────────────
    def run_gpio_triggered(self):
        """Block on GPIO thermal trigger (production mode)."""
        if not GPIO_AVAILABLE:
            print("[Node] GPIO not available — switching to demo loop")
            self.run_demo_loop()
            return

        print(f"[Node] Watching GPIO {THERMAL_TRIGGER_PIN} for thermal trigger …")
        try:
            while True:
                GPIO.wait_for_edge(THERMAL_TRIGGER_PIN, GPIO.RISING, timeout=5000)
                if GPIO.input(THERMAL_TRIGGER_PIN) == GPIO.HIGH:
                    self._on_trigger()
        except KeyboardInterrupt:
            pass
        finally:
            self.shutdown()

    def run_demo_loop(self, interval=5):
        """Capture every N seconds — for testing without hardware."""
        print(f"[Node] Demo loop — capture every {interval}s. Ctrl+C to stop.")
        try:
            while True:
                self._on_trigger()
                time.sleep(interval)
        except KeyboardInterrupt:
            pass
        finally:
            self.shutdown()

    def shutdown(self):
        print("\n[Node] Shutting down …")
        self.hb.stop()
        self.lora.deactivate()
        self.camera.stop()

        if GPIO_AVAILABLE:
            GPIO.output(STATUS_LED_PIN, GPIO.LOW)
            GPIO.cleanup()

        if self.detection_log:
            log_path = f"detection_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(log_path, "w") as f:
                json.dump(self.detection_log, f, indent=2)
            print(f"[Node] Log saved → {log_path}")

        self.lora.close()
        print("[Node] Done.")


# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Farm Animal Intrusion — Raspberry Pi 5 LoRa Node"
    )
    parser.add_argument("--model",     type=str,   default="mobilenet_farm.tflite")
    parser.add_argument("--cam",       type=int,   default=0)
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--simulate",  action="store_true",
                        help="Simulate LoRa output (no hardware needed)")
    parser.add_argument("--demo",      action="store_true",
                        help="Run timed demo loop instead of GPIO trigger")
    args = parser.parse_args()

    controller = NodeController(
        model_path     = args.model,
        cam_index      = args.cam,
        conf_threshold = args.threshold,
        simulate       = args.simulate,
    )
    controller.start()

    if args.demo or not GPIO_AVAILABLE:
        controller.run_demo_loop()
    else:
        controller.run_gpio_triggered()
