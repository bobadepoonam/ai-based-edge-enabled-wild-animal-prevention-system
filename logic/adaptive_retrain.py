#!/usr/bin/env python3

import os
import csv
import json
import shutil
import random
import logging
import argparse
import subprocess
import numpy as np
from pathlib import Path
from datetime import datetime

try:
    from ultralytics import YOLO
    import torch
except ImportError as e:
    print(f"[RETRAIN] Missing: {e} — run: pip install ultralytics torch")
    exit(1)

try:
    import tensorflow as tf
    from tensorflow import keras
except ImportError as e:
    print(f"[RETRAIN] TensorFlow missing: {e} — run: pip install tensorflow")
    exit(1)

BASE_DIR         = Path("/home/aivisionhub/wildlife_project")
CAPTURES_DIR     = BASE_DIR / "captures"
MODELS_DIR       = BASE_DIR / "models"
DATASET_DIR      = BASE_DIR / "retrain_dataset"
RETRAIN_LOG      = BASE_DIR / "retrain_log.csv"
STATE_FILE       = BASE_DIR / "retrain_state.json"

ACTIVE_YOLO      = BASE_DIR / "yolov8n_active.pt"
BASE_YOLO        = BASE_DIR / "yolov8n_base.pt"
MOBILENET_PATH   = BASE_DIR / "model2_mobilenet.keras"
MOBILENET_LABELS = BASE_DIR / "class_labels.txt"

MIN_NEW_IMAGES   = 50
MAX_IMAGES       = 500
YOLO_EPOCHS      = 15
MOBILENET_EPOCHS = 10
IMGSZ            = 320
VAL_SPLIT        = 0.15
MIN_MAP_GAIN     = 0.005
CONF_THRESHOLD   = 0.45

WATCHED_CLASSES = {
    "Person", "Man", "Woman", "Boy", "Girl", "Human body", "Human face", "Human head",
    "Elephant", "Bear", "Brown bear", "Tiger", "Lion", "Leopard",
    "Jaguar (Animal)", "Cheetah", "Rhinoceros", "Hippopotamus",
    "Pig", "Bull", "Cattle", "Camel", "Dog",
    "Goat", "Deer", "Antelope", "Horse", "Mule", "Sheep",
    "Kangaroo", "Alpaca", "Porcupine", "Lynx", "Ostrich", "Zebra",
    "Monkey", "Cat", "Fox", "Rabbit", "Squirrel", "Raccoon",
    "Hedgehog", "Panda", "Red panda", "Otter", "Koala", "Skunk",
    "Bird", "Chicken", "Duck", "Goose", "Parrot", "Owl",
    "Eagle", "Falcon", "Sparrow", "Magpie", "Raven", "Turkey", "Swan",
    "Snake", "Crocodile", "Lizard", "Tortoise", "Turtle",
    "Bee", "Ant", "Scorpion",
    "Car", "Truck", "Van", "Motorcycle", "Bus", "Taxi",
    "human", "cow", "bear", "deer", "goat", "wild boar",
}

logging.basicConfig(
    level=logging.INFO,
    format="[RETRAIN] %(asctime)s  %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("retrain")


def setup_dirs():
    for d in [MODELS_DIR,
              DATASET_DIR / "images" / "train",
              DATASET_DIR / "images" / "val",
              DATASET_DIR / "labels" / "train",
              DATASET_DIR / "labels" / "val"]:
        d.mkdir(parents=True, exist_ok=True)

    if not RETRAIN_LOG.exists():
        with open(RETRAIN_LOG, "w", newline="") as f:
            csv.writer(f).writerow([
                "Timestamp", "Run_ID",
                "Images_Used", "Train_Images", "Val_Images",
                "YOLO_Epochs", "MobileNet_Epochs",
                "YOLO_mAP_Before", "YOLO_mAP_After", "YOLO_mAP_Gain",
                "YOLO_Accepted",
                "MobileNet_Acc_Before", "MobileNet_Acc_After",
                "MobileNet_Accepted",
                "Active_YOLO", "Active_MobileNet", "Notes"
            ])

    if not BASE_YOLO.exists():
        log.info("Downloading base YOLOv8n weights...")
        m = YOLO("yolov8n.pt")
        shutil.copy(m.ckpt_path if hasattr(m, "ckpt_path") else "yolov8n.pt", BASE_YOLO)

    if not ACTIVE_YOLO.exists():
        shutil.copy(BASE_YOLO, ACTIVE_YOLO)


def load_state():
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"last_retrain_ts": None, "last_image_count": 0, "retrain_count": 0}


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def collect_detection_images():
    pairs = []
    for date_dir in sorted(CAPTURES_DIR.glob("*")):
        if not date_dir.is_dir():
            continue
        labels_dir = date_dir / "labels"
        if not labels_dir.exists():
            for img in date_dir.glob("*.jpg"):
                pairs.append((img, None))
            continue
        for lbl in labels_dir.glob("*.txt"):
            img = date_dir / (lbl.stem + ".jpg")
            if not img.exists():
                img = date_dir / (lbl.stem + ".png")
            if img.exists():
                pairs.append((img, lbl))
    log.info(f"Found {len(pairs)} detection images")
    return pairs


def filter_watched_classes(pairs, yolo_model):
    watched_ids = {
        cls_id for cls_id, name in yolo_model.names.items()
        if name.lower() in {c.lower() for c in WATCHED_CLASSES}
    }
    filtered = []
    for img_path, lbl_path in pairs:
        if lbl_path and lbl_path.exists():
            with open(lbl_path) as f:
                lines = f.readlines()
            cls_in_img = {int(l.split()[0]) for l in lines if l.strip()}
            if cls_in_img & watched_ids:
                filtered.append((img_path, lbl_path, lines))
        else:
            results = yolo_model(str(img_path), conf=CONF_THRESHOLD, verbose=False)
            lines = []
            has_watched = False
            for r in results:
                for box in r.boxes:
                    cid = int(box.cls[0])
                    if cid in watched_ids:
                        has_watched = True
                        x, y, w, hh = box.xywhn[0].tolist()
                        lines.append(f"{cid} {x:.6f} {y:.6f} {w:.6f} {hh:.6f}\n")
            if has_watched:
                filtered.append((img_path, None, lines))
    log.info(f"Filtered to {len(filtered)} images with watched classes")
    return filtered


def build_yolo_dataset(filtered_pairs, yolo_model):
    shutil.rmtree(DATASET_DIR, ignore_errors=True)
    for split in ["train", "val"]:
        (DATASET_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (DATASET_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    if len(filtered_pairs) > MAX_IMAGES:
        filtered_pairs = filtered_pairs[-MAX_IMAGES:]
    random.shuffle(filtered_pairs)

    n_val   = max(1, int(len(filtered_pairs) * VAL_SPLIT))
    n_train = len(filtered_pairs) - n_val

    val_set   = filtered_pairs[:n_val]
    train_set = filtered_pairs[n_val:]

    def copy_pair(img_path, lbl_path, label_lines, split):
        stem    = img_path.stem + f"_{split}_{img_path.parent.name}"
        dst_img = DATASET_DIR / "images" / split / (stem + img_path.suffix)
        dst_lbl = DATASET_DIR / "labels" / split / (stem + ".txt")
        shutil.copy(img_path, dst_img)
        if lbl_path and lbl_path.exists():
            shutil.copy(lbl_path, dst_lbl)
        elif label_lines:
            with open(dst_lbl, "w") as f:
                f.writelines(label_lines)

    for img, lbl, lines in train_set:
        copy_pair(img, lbl, lines, "train")
    for img, lbl, lines in val_set:
        copy_pair(img, lbl, lines, "val")

    watched_sorted = sorted(WATCHED_CLASSES)
    yaml_path = DATASET_DIR / "dataset.yaml"
    with open(yaml_path, "w") as f:
        f.write(f"path: {DATASET_DIR}\n")
        f.write("train: images/train\n")
        f.write("val:   images/val\n")
        f.write(f"nc: {len(watched_sorted)}\n")
        f.write(f"names: {watched_sorted}\n")

    log.info(f"YOLO dataset: {n_train} train, {n_val} val")
    return yaml_path, n_train, n_val


def build_mobilenet_dataset(filtered_pairs):
    import cv2
    X, y = [], []
    label_map = {}

    for img_path, _, label_lines in filtered_pairs:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img = cv2.resize(img, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        fname = img_path.stem.lower()
        matched = None
        for cls in WATCHED_CLASSES:
            if cls.lower() in fname:
                matched = cls.lower()
                break

        if matched is None and label_lines:
            try:
                cid = int(label_lines[0].split()[0])
                matched = f"class_{cid}"
            except Exception:
                pass

        if matched is None:
            continue

        if matched not in label_map:
            label_map[matched] = len(label_map)
        X.append(img.astype(np.float32) / 255.0)
        y.append(label_map[matched])

    if len(X) < 5:
        return None, None, None

    X = np.array(X)
    y = np.array(y)
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]
    n_val = max(1, int(len(X) * VAL_SPLIT))
    return (X[n_val:], y[n_val:], X[:n_val], y[:n_val],
            list(label_map.keys()))


def evaluate_model_yolo(model_path, yaml_path):
    try:
        m       = YOLO(str(model_path))
        metrics = m.val(data=str(yaml_path), imgsz=IMGSZ, conf=CONF_THRESHOLD, verbose=False)
        return float(metrics.box.map50)
    except Exception as e:
        log.warning(f"YOLO eval failed: {e}"); return 0.0


def evaluate_model_mobilenet(model_path, X_val, y_val):
    try:
        m    = keras.models.load_model(str(model_path))
        loss, acc = m.evaluate(X_val, y_val, verbose=0)
        return acc
    except Exception as e:
        log.warning(f"MobileNet eval failed: {e}"); return 0.0


def fine_tune_yolo(yaml_path, run_id):
    log.info(f"Fine-tuning YOLOv8n — run {run_id}")
    try:
        yolo = YOLO(str(ACTIVE_YOLO))
        yolo.train(
            data          = str(yaml_path),
            epochs        = YOLO_EPOCHS,
            imgsz         = IMGSZ,
            batch         = 8,
            lr0           = 0.001,
            lrf           = 0.01,
            momentum      = 0.937,
            weight_decay  = 0.0005,
            warmup_epochs = 2,
            project       = str(MODELS_DIR),
            name          = f"{run_id}_yolo",
            exist_ok      = True,
            verbose       = False,
            device        = "cpu",
            workers       = 2,
            cache         = False,
            save          = True,
            save_period   = -1,
        )
        best = MODELS_DIR / f"{run_id}_yolo" / "weights" / "best.pt"
        return best if best.exists() else None
    except Exception as e:
        log.error(f"YOLO fine-tune failed: {e}"); return None


def fine_tune_mobilenet(X_train, y_train, X_val, y_val, num_classes, run_id):
    log.info(f"Fine-tuning MobileNetV2 — run {run_id}")
    try:
        if MOBILENET_PATH.exists():
            model = keras.models.load_model(str(MOBILENET_PATH))
            log.info("Loaded existing MobileNetV2 for fine-tuning")
            for layer in model.layers[:-30]:
                layer.trainable = False
        else:
            base = keras.applications.MobileNetV2(
                input_shape=(224, 224, 3), include_top=False, weights="imagenet")
            base.trainable = False
            inputs  = keras.Input(shape=(224, 224, 3))
            x       = keras.applications.mobilenet_v2.preprocess_input(inputs * 255.0)
            x       = base(x, training=False)
            x       = keras.layers.GlobalAveragePooling2D()(x)
            x       = keras.layers.Dense(256, activation="relu")(x)
            x       = keras.layers.Dropout(0.4)(x)
            outputs = keras.layers.Dense(num_classes, activation="softmax")(x)
            model   = keras.Model(inputs, outputs)
            log.info("Built new MobileNetV2 (no existing model found)")

        model.compile(
            optimizer=keras.optimizers.Adam(1e-4),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=MOBILENET_EPOCHS,
            batch_size=16,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor="val_accuracy", patience=4,
                    restore_best_weights=True)
            ],
            verbose=0
        )
        new_path = MODELS_DIR / f"{run_id}_mobilenet.keras"
        model.save(str(new_path))
        log.info(f"MobileNetV2 saved: {new_path}")
        return new_path
    except Exception as e:
        log.error(f"MobileNet fine-tune failed: {e}"); return None


def log_retrain(run_id, n_images, n_train, n_val,
                yolo_map_before, yolo_map_after, yolo_map_gain, yolo_accepted,
                mn_acc_before, mn_acc_after, mn_accepted,
                active_yolo, active_mn, notes=""):
    with open(RETRAIN_LOG, "a", newline="") as f:
        csv.writer(f).writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            run_id, n_images, n_train, n_val,
            YOLO_EPOCHS, MOBILENET_EPOCHS,
            f"{yolo_map_before:.4f}", f"{yolo_map_after:.4f}", f"{yolo_map_gain:+.4f}",
            "YES" if yolo_accepted else "NO",
            f"{mn_acc_before:.4f}", f"{mn_acc_after:.4f}",
            "YES" if mn_accepted else "NO",
            str(active_yolo), str(active_mn), notes
        ])


def run_retrain(force=False):
    setup_dirs()
    state  = load_state()
    run_id = datetime.now().strftime("retrain_%Y%m%d_%H%M%S")

    log.info("=" * 52)
    log.info(f" Adaptive Retrain — {run_id}")
    log.info("=" * 52)

    all_pairs = collect_detection_images()
    n_total   = len(all_pairs)
    new_images = n_total - state["last_image_count"]
    log.info(f"Total: {n_total}  New: {new_images}  Trigger: {MIN_NEW_IMAGES}")

    if not force and new_images < MIN_NEW_IMAGES:
        log.info(f"Not enough new images ({new_images} < {MIN_NEW_IMAGES}). Skipping.")
        return

    log.info("Loading active YOLO for class filtering...")
    active_yolo = YOLO(str(ACTIVE_YOLO))
    filtered    = filter_watched_classes(all_pairs, active_yolo)

    if len(filtered) < 10:
        log.warning(f"Too few usable images ({len(filtered)}). Skipping.")
        return

    yaml_path, n_train, n_val = build_yolo_dataset(filtered, active_yolo)

    log.info("Evaluating current YOLO (baseline)...")
    yolo_map_before = evaluate_model_yolo(ACTIVE_YOLO, yaml_path)

    new_yolo = fine_tune_yolo(yaml_path, run_id)
    yolo_map_after  = 0.0
    yolo_accepted   = False
    if new_yolo:
        yolo_map_after = evaluate_model_yolo(new_yolo, yaml_path)
        yolo_map_gain  = yolo_map_after - yolo_map_before
        yolo_accepted  = yolo_map_gain >= MIN_MAP_GAIN
        if yolo_accepted:
            archive = MODELS_DIR / f"yolov8n_v{state['retrain_count']:03d}.pt"
            shutil.copy(ACTIVE_YOLO, archive)
            shutil.copy(new_yolo, ACTIVE_YOLO)
            log.info(f"YOLO ACCEPTED (+{yolo_map_gain:.4f}) → {ACTIVE_YOLO}")
        else:
            log.info(f"YOLO REJECTED (gain={yolo_map_gain:+.4f} < {MIN_MAP_GAIN})")
    else:
        yolo_map_gain = 0.0

    mn_acc_before = 0.0
    mn_acc_after  = 0.0
    mn_accepted   = False
    active_mn     = MOBILENET_PATH

    mobilenet_data = build_mobilenet_dataset(filtered)
    if mobilenet_data and mobilenet_data[0] is not None:
        X_train, y_train, X_val, y_val, mn_classes = mobilenet_data
        num_cls = len(mn_classes)
        log.info(f"MobileNet dataset: {len(X_train)} train, {len(X_val)} val, {num_cls} classes")

        if MOBILENET_PATH.exists():
            original_mn = keras.models.load_model(str(MOBILENET_PATH))
            mn_acc_before = float(original_mn.evaluate(X_val, y_val, verbose=0)[1])
            log.info(f"MobileNet baseline accuracy: {mn_acc_before:.4f}")

        new_mn = fine_tune_mobilenet(X_train, y_train, X_val, y_val, num_cls, run_id)

        if new_mn:
            mn_acc_after = evaluate_model_mobilenet(new_mn, X_val, y_val)
            mn_gain      = mn_acc_after - mn_acc_before
            mn_accepted  = mn_gain >= 0.001

            if mn_accepted:
                if MOBILENET_PATH.exists():
                    archive_mn = MODELS_DIR / f"mobilenet_v{state['retrain_count']:03d}.keras"
                    shutil.copy(MOBILENET_PATH, archive_mn)
                shutil.copy(new_mn, MOBILENET_PATH)
                active_mn = MOBILENET_PATH
                log.info(f"MobileNet ACCEPTED (+{mn_gain:.4f}) → {MOBILENET_PATH}")

                with open(BASE_DIR / "class_labels.txt", "w") as f:
                    f.write("\n".join(mn_classes))
                log.info("class_labels.txt updated")
            else:
                log.info(f"MobileNet REJECTED (gain={mn_gain:+.4f})")

    log_retrain(
        run_id, len(filtered), n_train, n_val,
        yolo_map_before, yolo_map_after, yolo_map_gain if new_yolo else 0.0, yolo_accepted,
        mn_acc_before, mn_acc_after, mn_accepted,
        ACTIVE_YOLO, active_mn,
        notes=f"YOLO: {'+' if yolo_accepted else '-'}  MobileNet: {'+' if mn_accepted else '-'}"
    )

    state["last_retrain_ts"]  = datetime.now().isoformat()
    state["last_image_count"] = n_total
    state["retrain_count"]   += 1
    save_state(state)

    shutil.rmtree(DATASET_DIR, ignore_errors=True)

    log.info(f"Retrain #{state['retrain_count']} complete.")
    log.info(f"YOLO mAP@50: {yolo_map_before:.4f} → {yolo_map_after:.4f}")
    log.info(f"MobileNet acc: {mn_acc_before:.4f} → {mn_acc_after:.4f}")
    log.info(f"YOLO accepted: {'YES' if yolo_accepted else 'NO'}  "
             f"MobileNet accepted: {'YES' if mn_accepted else 'NO'}")
    log.info("=" * 52)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--min-images", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    if args.min_images:
        MIN_NEW_IMAGES = args.min_images
    if args.epochs:
        YOLO_EPOCHS = args.epochs

    run_retrain(force=args.force)
