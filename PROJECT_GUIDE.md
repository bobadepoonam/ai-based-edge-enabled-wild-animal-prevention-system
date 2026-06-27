# AI-Based Edge-Enabled Adaptive Farm Animal Intrusion Prevention System
### PhD Research — Complete Technical Reference

---

## Abstract

A fully autonomous, solar-powered, wireless edge-AI system for farm perimeter protection. The system uses a two-node architecture: a Raspberry Pi 5 master node running a dual-model AI pipeline (YOLOv8n + MobileNetV2), and an ESP32 actuator node controlling deterrents (110 dB siren, high-intensity lights, IR lights). The nodes communicate wirelessly over LoRa (433 MHz, ~1 km range) with no internet dependency. The AI pipeline adapts continuously — an RL adapter tunes confidence thresholds in real-time, and a nightly retraining job fine-tunes both models on field-captured detections. The entire system runs on a 20W solar panel with a 5200 mAh Li-ion battery backup, making it fully off-grid suitable.

---

## Table of Contents

1. [Hardware Components](#1-hardware-components)
2. [System Architecture](#2-system-architecture)
3. [Power System](#3-power-system)
4. [Power Consumption Analysis](#4-power-consumption-analysis)
5. [Wireless Communication (LoRa)](#5-wireless-communication-lora)
6. [AI & Software Stack](#6-ai--software-stack)
7. [Algorithms Used](#7-algorithms-used)
8. [Day / Night Deterrent Logic](#8-day--night-deterrent-logic)
9. [Adaptive Learning](#9-adaptive-learning)
10. [File Structure](#10-file-structure)
11. [Setup & Deployment](#11-setup--deployment)
12. [Research Figures Guide](#12-research-figures-guide)
13. [Troubleshooting](#13-troubleshooting)

---

## 1. Hardware Components

### Master Node — Raspberry Pi 5

| Component | Spec | Purpose |
|-----------|------|---------|
| **Raspberry Pi 5** | 4 GB RAM, Broadcom BCM2712, quad-core A76 @ 2.4 GHz | Main AI inference node |
| **RPi Camera Module v2** | Sony IMX219, 8 MP, 1080p30 / 720p60, CSI interface | RGB image capture for AI classification |
| **MLX90640 Thermal Camera** | 32×24 pixel array, I2C, −40 to 300°C range, 2 Hz | Stage 1 warm-body detection |
| **IR Lights (850 nm)** | 850nm IR LED array, GPIO-controlled via relay | Night illumination for camera — connected to Pi hub |
| **LoRa Module SX1276** | 433 MHz, SPI bus 0, GPIO 23 RST, ~1 km LOS range | Wireless command TX to ESP32 |
| **Power Adapter** | 5V / 5A USB-C PD (27W) Official RPi adapter | Primary mains power for Pi 5 hub |
| microSD Card | 64 GB Class 10 A2 | OS, models, dataset, logs |

### Actuator Node — ESP32

| Component | Spec | Purpose |
|-----------|------|---------|
| **ESP32 Dev Module** | Dual-core Xtensa LX6 @ 240 MHz, 520 KB SRAM | LoRa receive, relay control, watchdog |
| **LoRa Module SX1276** | 433 MHz, SPI, GPIO 5 CS / GPIO 14 RST / GPIO 2 DIO0 | Wireless command RX from Pi 5 |
| **110 dB Siren** | 12V DC, relay-controlled (GPIO 26) | High-decibel animal deterrent |
| **High-Intensity White Lights** | 12V LED flood, relay-controlled (GPIO 25) | Night-time visual deterrent |
| **Relay Module** | 4-channel 5V relay, optocoupler isolated | Controls siren and lights from ESP32 GPIO |

### Power System

| Component | Spec | Purpose |
|-----------|------|---------|
| **Solar Panel** | 20W monocrystalline, Voc 22V, Vmp 18V | Primary power source (daylight hours) |
| **Li-ion Battery Pack** | 5200 mAh, 11.1V (3S), 57.7 Wh | Backup for night / overcast periods |
| **Solar Charge Controller** | PWM or MPPT, 12V/24V auto, 10A | Manages solar → battery charging |
| **Buck Converter (ESP32)** | Input 9–15V, Output 5V / 1A | Battery → ESP32 + relay board |

---

## 2. System Architecture

```
                    ┌──────────────────────────────────────────────┐
                    │          MASTER NODE — Raspberry Pi 5        │
                    │                                              │
  [Solar/Battery]   │  Stage 1: MLX90640 Thermal Camera           │
       │            │  ├─ Reads 768 pixels @ 2 Hz                 │
  [Buck→5V]─────→   │  ├─ Hot-pixel threshold: ≥20 px > 30°C      │
                    │  └─ Triggers Stage 2 on anomaly             │
                    │                                              │
                    │  Stage 2: RPi Camera v2 + Dual AI Model      │
                    │  ├─ YOLOv8n  → object detection             │
                    │  ├─ MobileNetV2 → classification             │
                    │  ├─ Fused decision (confidence-weighted)     │
                    │  └─ Matched against DETERRENT_MAP            │
                    │                                              │
                    │  LoRa SX1276 (SPI0, GPIO23 RST)             │
                    └────────────────────┬─────────────────────────┘
                                         │  433 MHz LoRa (~1 km)
                    ┌────────────────────▼─────────────────────────┐
                    │          ACTUATOR NODE — ESP32                │
                    │                                              │
                    │  LoRa SX1276 RX → parse command             │
                    │  GPIO 26 → Relay → 110 dB Siren             │
                    │  GPIO 25 → Relay → High-Intensity Lights     │
                    │  GPIO 27 → Relay → IR Lights                 │
                    │  Watchdog: all OFF if no heartbeat 2 min     │
                    └──────────────────────────────────────────────┘
```

### Communication Protocol

- Pi 5 sends heartbeat every **30 seconds** over LoRa
- On detection, Pi 5 sends deterrent command: `ACTIVATE_SIREN`, `ACTIVATE_LIGHTS`, `ACTIVATE_BOTH`
- On animal departure (3 missed frames), Pi 5 sends: `DEACTIVATE_*`
- LoRa sync word: `0x12` (must match both ends)
- LoRa frequency: `433 MHz` (change to 868 MHz EU / 915 MHz USA in both files)

---

## 3. Power System

```
☀ SOLAR + BATTERY (Edge Node only)
    │
[Charge Controller] → [5200mAh Li-ion Battery]
    │
[Buck → 5V/1A] → [ESP32 + Relays] → [110dB Siren / Hi-Intensity Lights]

🔌 MAINS ADAPTER (Hub only)
[Official 5V/5A USB-C] → [Raspberry Pi 5] → [MLX90640 / RPi Cam v2 / IR Lights / LoRa TX]
```

### Solar Autonomy Calculation

| Parameter | Value |
|-----------|-------|
| Solar panel output | 20W × 5 peak sun hours = **100 Wh/day** |
| Pi 5 idle consumption | 4W × 24h = 96 Wh/day |
| Pi 5 under AI load | ~8W (peaks during inference) |
| ESP32 + relay standby | ~0.5W |
| Siren (when active) | ~12W (intermittent) |
| Battery capacity | 5200 mAh × 11.1V = **57.7 Wh** |
| Battery backup duration (Pi idle) | 57.7 Wh ÷ 4W ≈ **~14 hours** |

> **Verdict:** 20W solar is sufficient for continuous EDGE NODE operation in regions with ≥5 peak sun hours/day. Battery covers the night. In low-sun periods, reduce inference frequency via thermal threshold tuning.

---

## 4. Power Consumption Analysis

| Component | Idle | Active | Notes |
|-----------|------|--------|-------|
| Raspberry Pi 5 (OS idle) | 2.7W | — | Quad-core A76, measured |
| Pi 5 + thermal read loop | 4.0W | — | Stage 1 running |
| Pi 5 + YOLOv8 inference | — | 7–9W | ~10 s burst per detection |
| Pi 5 + MobileNetV2 | — | 6–8W | Slightly lighter than YOLO |
| LoRa SX1276 TX | — | ~120 mA @ 3.3V = 0.4W | Per 30s heartbeat pulse |
| MLX90640 | 9mA @ 3.3V | — | Always on, negligible |
| RPi Camera v2 | ~250mA @ 5V | — | Active only during capture |
| ESP32 (WiFi off) | 30–80 mA @ 5V = 0.15–0.4W | — | Deep-sleep possible |
| 110dB Siren | — | ~1A @ 12V = 12W | Relay-switched, intermittent |
| LED Lights (Hi-power) | — | ~2–3A @ 12V = 24–36W | Relay-switched |
| IR Lights | — | ~0.5A @ 12V = 6W | Night only |

### Power-Saving Strategies Implemented

- **Stage 1 first:** Thermal camera runs continuously at 2 Hz (negligible power). RPi Camera and AI inference only activate on thermal trigger.
- **Inference bursts only:** YOLOv8 + MobileNetV2 run only during active sessions, not continuously.
- **IDLE_INTERVAL = 0.5s:** Pi sleeps 0.5s between thermal reads — CPU idles between polls.
- **ESP32 deep sleep (optional):** ESP32 can sleep between LoRa receive windows to save power.
- **Heartbeat-only LoRa:** LoRa TX only during 30s heartbeat pulses and detection events.
- **Deterrents relay-controlled:** Zero standby power draw for siren/lights.

---

## 5. Wireless Communication (LoRa)

| Parameter | Value |
|-----------|-------|
| Module | SX1276 (Semtech) |
| Frequency | 433 MHz (ISM band) |
| Modulation | LoRa (CSS — Chirp Spread Spectrum) |
| Range (LOS) | ~1 km typical, up to 2 km with antenna |
| Range (NLOS) | 200–500 m through vegetation |
| Bandwidth | 125 kHz |
| Spreading Factor | SF7 (default) — SF12 for max range |
| Sync Word | 0x12 (private network) |
| TX Power | 17 dBm (max for SX1276) |
| Interface (Pi) | SPI bus 0, CS0, RST = GPIO 23 |
| Interface (ESP32) | SPI, CS = GPIO 5, RST = GPIO 14, DIO0 = GPIO 2 |

### Message Types

| Message | Sender | Receiver | When |
|---------|--------|----------|------|
| `HEARTBEAT` | Pi 5 | ESP32 | Every 30 seconds |
| `ACTIVATE_SIREN` | Pi 5 | ESP32 | Animal detected (siren / both mode) |
| `DEACTIVATE_SIREN` | Pi 5 | ESP32 | Animal left |
| `ACTIVATE_LIGHTS` | Pi 5 | ESP32 | Animal detected (lights / both mode, night) |
| `DEACTIVATE_LIGHTS` | Pi 5 | ESP32 | Animal left |
| `ACTIVATE_BOTH` | Pi 5 | ESP32 | High-priority animal at night |
| `DEACTIVATE_BOTH` | Pi 5 | ESP32 | Animal left |

### Watchdog Failsafe

If ESP32 receives no `HEARTBEAT` for **2 minutes** (Pi 5 crash / power cut), it forces all relays OFF. This prevents indefinite siren/light activation.

---

## 6. AI & Software Stack

| Layer | Technology |
|-------|-----------|
| OS | Raspberry Pi OS Lite (64-bit) |
| Language | Python 3.11 |
| Object Detection | YOLOv8n (Ultralytics) — 3.2M params, COCO pretrained |
| Classification | MobileNetV2 (TensorFlow/Keras) — ImageNet pretrained, fine-tuned |
| RL Adaptation | REINFORCE (Policy Gradient) — custom TF implementation |
| Thermal | adafruit-circuitpython-mlx90640 |
| Camera | rpicam-still (Pi 5 camera stack) |
| LoRa | spidev + lgpio (bare-metal SX1276 register writes) |
| ESP32 Firmware | Arduino IDE + LoRa by Sandeep Mistry |
| Inference Export | TFLite int8 (MobileNetV2 for Pi deployment) |

---

## 7. Algorithms Used

### Model 1 — Custom CNN + REINFORCE RL
- 4-block VGG-style CNN: Conv→BN→ReLU→MaxPool × 4 → GAP → Dense
- RL Policy: REINFORCE (Monte Carlo Policy Gradient)
- State: softmax probability vector from CNN
- Action: class selection (alert / suppress)
- Reward: +1 correct, −1 false alarm, 0 uncertain
- Update: gradient ascent on log-prob weighted by discounted return

### Model 2 — MobileNetV2 + REINFORCE RL
- MobileNetV2 ImageNet backbone (depthwise separable convolutions)
- Two-phase training: frozen backbone (Phase 1) → unfreeze top 30 layers (Phase 2)
- Same REINFORCE RL adapter for threshold tuning
- **Deployed on Pi 5 as primary classifier**

### Model 3 — YOLOv8-Style + Threshold RL
- CSP (Cross-Stage Partial) backbone, depthwise + pointwise convolutions
- RL: per-class confidence threshold adaptation
- Threshold raised for high-FP classes, lowered for high-FN classes
- Exponential decay factor 0.95 on error rates

### Production Models
- **YOLOv8n** — object detection backbone (prod, retrained nightly)
- **MobileNetV2** — classification (prod, retrained nightly)
- Both run on every detection frame and results are **fused**

### Fusion Logic
```
if both agree → use that label, max(confidence)
if MobileNet confidence > YOLO confidence + 15% → use MobileNet
else → use YOLO
```

### Adaptive Learning
- **RL Adapter:** Runs real-time on every detection, adjusts thresholds in RAM
- **Nightly retrain:** fine-tunes both YOLOv8n + MobileNetV2 on confirmed field captures, triggered by cron at 2 AM OR when 50 confirmed intrusions accumulate

---

## 8. Day / Night Deterrent Logic

| Time | Human | Lights-only animals | Siren-only animals | Both-mode animals |
|------|-------|--------------------|--------------------|-------------------|
| **Day (6AM–7PM)** | Log only | Log only | Siren only | Siren only |
| **Night (7PM–6AM)** | Lights only | Lights | Siren | Siren + Lights |

**Rationale:**
- Daytime human = likely farm worker → no deterrent, just log
- Night human = intruder → lights to reveal them without alerting that the system is AI-based
- Daytime lights are ineffective as a deterrent (visibility too high) → siren only
- Night = full deterrent capability activated

---

## 9. Adaptive Learning

### Layer 1 — RL Adapter (Real-Time)
- Runs on every single detection frame
- Adjusts per-class confidence thresholds in memory
- Zero disk writes, zero retraining
- Decay factor 0.95 ensures old mistakes fade over time

### Layer 2 — Triggered Retraining
**Trigger:** Either of these conditions:
- 50 confirmed intrusions logged (`intrusion_count.json`)
- Cron job at 2:00 AM daily

**Process:**
1. Scan `captures/` for all confirmed detection images
2. Filter to watched classes only
3. Build YOLO-format + MobileNet dataset from captures
4. Fine-tune **YOLOv8n** (15 epochs, CPU, batch 8)
5. Fine-tune **MobileNetV2** (10 epochs, unfreeze top 30 layers)
6. Evaluate both: accept only if improvement ≥ 0.5% mAP / accuracy
7. Log results to `retrain_log.csv`
8. Reload model in master process (without restart)

---

## 10. File Structure

```
files/
├── phd_research_notebook.ipynb    ← Main PhD notebook
├── PROJECT_GUIDE.html             ← Beautiful HTML guide (open in browser)
├── PROJECT_GUIDE.md               ← This markdown guide
│
├── model1_custom_cnn.py           ← Custom CNN + RL
├── model2_mobilenet.py            ← MobileNetV2 + RL
├── model3_yolo.py                 ← YOLOv8-Style + RL
├── dataset_utils.py               ← Dataset loading & preprocessing
├── metrics_utils.py               ← Evaluation + chart generation
│
├── logic/
│   ├── farm_intrusion_master.py   ← Pi 5 production script
│   ├── adaptive_retrain.py        ← Triggered/nightly retraining
│   └── esp32_wildlife_listener.ino
│
├── dataset/                       ← Training data (6 classes, 5200+ images)
│   ├── bear/  cow/  deer/  goat/  human/  wild boar/
│
├── test/                          ← Custom test images (flat folder)
│
├── saved_models/
│   ├── model1_custom_cnn.keras
│   ├── model2_mobilenet.keras
│   └── model3_yolo_style.keras
│
└── results/
    ├── figures/     ← 16 PNG research charts
    ├── logs/        ← CSV result files
    └── tflite/      ← Quantized model for Pi 5
```

---

## 11. Setup & Deployment

### Raspberry Pi 5 Setup

```bash
sudo apt update && sudo apt install -y python3-pip python3-venv libopencv-dev
sudo raspi-config   # Enable SPI, I2C, Camera

python3 -m venv ~/venv && source ~/venv/bin/activate

pip install tensorflow-lite-runtime opencv-python-headless numpy \
            ultralytics spidev lgpio adafruit-blinka \
            adafruit-circuitpython-mlx90640
```

### Copy Models to Pi

```powershell
# From Windows:
scp saved_models\model2_mobilenet.keras pi@raspberrypi.local:/home/pi/farm/
scp results\tflite\best_model_int8.tflite pi@raspberrypi.local:/home/pi/farm/
scp logic\farm_intrusion_master.py pi@raspberrypi.local:/home/pi/farm/
scp logic\adaptive_retrain.py pi@raspberrypi.local:/home/pi/farm/
```

### Run Master Script

```bash
cd /home/pi/farm
source ~/venv/bin/activate
python3 farm_intrusion_master.py
```

### Cron — Nightly Retraining (2 AM)

```bash
crontab -e
# Add:
0 2 * * * /home/pi/venv/bin/python3 /home/pi/farm/adaptive_retrain.py >> /home/pi/farm/retrain_cron.log 2>&1
```

### ESP32 Firmware

1. Arduino IDE → Install **LoRa by Sandeep Mistry**
2. Board: **ESP32 Dev Module**
3. Open `logic/esp32_wildlife_listener.ino`
4. Upload

---

## 12. Research Figures Guide

| Fig | File | For Paper |
|-----|------|-----------|
| 1 | fig01_class_distribution.png | Dataset balance |
| 2 | fig02_sample_images.png | Dataset diversity |
| 3–5 | fig03/04/05_training_curves.png | Convergence behaviour |
| 6–8 | fig06/07/08_confusion_*.png | Classification errors |
| 9 | fig09_metric_comparison.png | Accuracy/F1/mAP comparison |
| 10 | fig10_per_class_f1_heatmap.png | Per-class F1 |
| 11 | fig11_per_class_ap_heatmap.png | Per-class AP |
| 12 | fig12_precision_recall_curves.png | PR trade-off |
| 13 | fig13_roc_auc_curves.png | ROC-AUC |
| 14 | fig14_speed_vs_accuracy.png | Edge deployment trade-off |
| 15 | fig15_model_size_speed.png | Size / latency / edge score |
| 16 | fig16_radar_chart.png | Holistic radar comparison |

---

## 13. Troubleshooting

| Error | Fix |
|-------|-----|
| `UnicodeEncodeError` | `python -X utf8 script.py` |
| GPIO 23 busy | `sudo pinctrl set 23 ip` → rerun |
| LoRa not communicating | Check frequency, sync word 0x12, pin wiring |
| MobileNet not found on Pi | Copy `.keras` file, check `MOBILENET_PATH` |
| TFLite import error | `pip install tensorflow-lite-runtime` |
| MLX90640 I2C error | `sudo raspi-config` → enable I2C |
| Camera capture fail | `sudo raspi-config` → enable Camera, check ribbon |
| Low solar charging | Check charge controller, clean panel, check cable voltage drop |
| Battery draining too fast | Increase `IDLE_INTERVAL`, reduce `BASELINE_FRAMES` |

---

*AI-Based Edge-Enabled Adaptive Farm Animal Intrusion Prevention System — PhD Research 2025*
