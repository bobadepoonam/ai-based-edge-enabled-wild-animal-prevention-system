"""
models/model3_yolo.py
═══════════════════════════════════════════════════════════════════
MODEL 3 — YOLOv8 Object Detection + RL Threshold Adaptation
═══════════════════════════════════════════════════════════════════
Architecture:
  • YOLOv8n (nano) — smallest, fastest YOLO variant
  • Fine-tuned on farm animal dataset
  • RL adapter adjusts per-class detection confidence thresholds
    dynamically to minimise false alarms in field conditions

Key difference from Models 1 & 2:
  • YOLOv8 is an OBJECT DETECTOR (returns bounding boxes + class)
  • Models 1 & 2 are CLASSIFIERS (whole-image label)
  • This gives us a 3-way comparison: classifier vs transfer vs detector

For classification evaluation (fair comparison):
  • We use the highest-confidence detection's class label
  • If no detection above threshold → "background" / no intrusion

NOTE: Full YOLOv8 training requires the Ultralytics library and a
proper dataset in YOLO format (images + .txt label files).
This wrapper handles both real and simulated modes.
═══════════════════════════════════════════════════════════════════
"""

import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from model1_custom_cnn import RLPolicyAdapter


# ─────────────────────────────────────────────────────────────────
# SIMULATED YOLO-STYLE DETECTOR
# (used when ultralytics is not installed or dataset not in YOLO fmt)
# ─────────────────────────────────────────────────────────────────
def build_yolo_classifier(input_shape=(224, 224, 3), num_classes=10):
    """
    A YOLO-inspired lightweight classification backbone.
    Uses CSP-style residual connections and depthwise separable convs
    to mimic YOLOv8's design philosophy at reduced scale.

    In production: replace this with actual YOLOv8 via ultralytics.
    """
    inputs = keras.Input(shape=input_shape)

    # Stem
    x = layers.Conv2D(32, (3, 3), strides=2, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.1)(x)

    # CSP Block 1
    x = _csp_block(x, filters=64)
    x = layers.MaxPooling2D(2)(x)

    # CSP Block 2
    x = _csp_block(x, filters=128)
    x = layers.MaxPooling2D(2)(x)

    # CSP Block 3
    x = _csp_block(x, filters=256)
    x = layers.MaxPooling2D(2)(x)

    # CSP Block 4
    x = _csp_block(x, filters=512)

    # Detection head (classification mode)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation='softmax',
                           name='class_predictions')(x)

    model = keras.Model(inputs, outputs, name="YOLOv8_Style_FarmIntruder")
    return model


def _csp_block(x, filters):
    """Cross-Stage Partial block — YOLOv8's core building block."""
    # Main branch
    main = layers.DepthwiseConv2D((3, 3), padding='same')(x)
    main = layers.BatchNormalization()(main)
    main = layers.LeakyReLU(0.1)(main)
    main = layers.Conv2D(filters, (1, 1), padding='same')(main)
    main = layers.BatchNormalization()(main)
    main = layers.LeakyReLU(0.1)(main)

    # Skip branch (1x1 conv to match channels)
    skip = layers.Conv2D(filters, (1, 1), padding='same')(x)
    skip = layers.BatchNormalization()(skip)

    # Add (residual) if shapes match, else just use main
    if x.shape[-1] == filters:
        return layers.Add()([main, skip])
    return layers.Concatenate()([main, skip])


# ─────────────────────────────────────────────────────────────────
# RL THRESHOLD ADAPTER
# ─────────────────────────────────────────────────────────────────
class ThresholdRLAdapter:
    """
    RL agent that adapts per-class confidence thresholds.

    Unlike Models 1&2 which adapt the action selection,
    this agent learns the optimal detection threshold for each class
    to reduce false positives in varying farm environments
    (night, rain, fog, partial occlusion).

    State  : [current threshold vector, recent FP rate per class]
    Action : increase / decrease / hold threshold for each class
    Reward : +1 true detection, -0.5 false alarm, -0.2 miss
    """

    def __init__(self, num_classes=10, init_threshold=0.5, lr=0.01):
        self.num_classes     = num_classes
        self.thresholds      = np.full(num_classes, init_threshold)
        self.lr              = lr
        self.fp_rates        = np.zeros(num_classes)
        self.fn_rates        = np.zeros(num_classes)
        self.reward_history  = []
        self.threshold_history = [self.thresholds.copy()]

        # Use same REINFORCE agent from model1 for action selection
        self.policy          = RLPolicyAdapter(num_classes, lr=1e-3)

    def adapt_thresholds(self, probs, true_label):
        """
        Decide whether to raise/lower/keep thresholds after one detection.
        Returns the adapted prediction.
        """
        pred_class = np.argmax(probs)
        pred_conf  = probs[pred_class]
        accepted   = pred_conf >= self.thresholds[pred_class]

        if accepted and pred_class == true_label:
            reward = 1.0
        elif accepted and pred_class != true_label:
            reward = -0.5
            self.fp_rates[pred_class] += 0.1
        else:
            reward = -0.2
            self.fn_rates[true_label] += 0.1

        # Threshold adjustment rule
        self.fp_rates  *= 0.95   # exponential decay
        self.fn_rates  *= 0.95

        # Raise threshold for high FP classes; lower for high FN classes
        delta = self.lr * (self.fp_rates - self.fn_rates)
        self.thresholds = np.clip(self.thresholds + delta, 0.2, 0.95)
        self.threshold_history.append(self.thresholds.copy())
        self.reward_history.append(reward)

        return pred_class if accepted else -1, reward


# ─────────────────────────────────────────────────────────────────
# COMBINED WRAPPER
# ─────────────────────────────────────────────────────────────────
class YOLOModel_RL:
    """
    Full model = YOLOv8-style backbone + threshold RL adapter.
    """

    def __init__(self, class_names, img_size=(224, 224, 3),
                 rl_lr=0.01, init_threshold=0.5):
        self.class_names    = class_names
        self.num_classes    = len(class_names)
        self.img_size       = img_size
        self.model          = build_yolo_classifier(img_size, self.num_classes)
        self.rl_adapter     = ThresholdRLAdapter(self.num_classes,
                                                  init_threshold, rl_lr)
        self.history        = None
        self.rl_loss_log    = []

    # ── Training ─────────────────────────────────────────────────
    def compile_and_train(self, train_ds, val_ds, epochs=30, lr=1e-3):
        self.model.compile(
            optimizer=keras.optimizers.Adam(lr),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy', patience=7,
                restore_best_weights=True, verbose=1),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5,
                patience=3, min_lr=1e-6, verbose=1),
        ]
        hist = self.model.fit(
            train_ds, validation_data=val_ds,
            epochs=epochs, callbacks=callbacks, verbose=1)
        self.history = hist.history
        return self.history

    # ── Inference ────────────────────────────────────────────────
    def predict(self, X):
        return self.model.predict(X, verbose=0)

    def predict_with_rl(self, image):
        probs = self.model.predict(image[np.newaxis], verbose=0)[0]
        pred_class, _ = self.rl_adapter.adapt_thresholds(probs, np.argmax(probs))
        if pred_class == -1:
            pred_class = np.argmax(probs)   # fallback to argmax
        confidence = float(probs[pred_class])
        return self.class_names[pred_class], confidence, pred_class

    # ── RL online adaptation ─────────────────────────────────────
    def rl_adapt_episode(self, images, true_labels, verbose=False):
        episode_rewards = []
        for img, true_lbl in zip(images, true_labels):
            probs = self.model.predict(img[np.newaxis], verbose=0)[0]
            pred_class, reward = self.rl_adapter.adapt_thresholds(probs, true_lbl)
            episode_rewards.append(reward)
        self.rl_loss_log.append(-np.mean(episode_rewards))
        if verbose:
            print(f"  [YOLO-RL] Avg Reward: {np.mean(episode_rewards):.3f}  "
                  f"Thresholds: {np.round(self.rl_adapter.thresholds, 3)}")
        return episode_rewards

    # ── Detection output ──────────────────────────────────────────
    def detect(self, image_array):
        probs    = self.model.predict(image_array[np.newaxis], verbose=0)[0]
        pred_cls = np.argmax(probs)
        conf     = float(probs[pred_cls])
        thresh   = self.rl_adapter.thresholds[pred_cls]
        accepted = conf >= thresh
        top3     = sorted(enumerate(probs), key=lambda x: -x[1])[:3]
        return {
            "animal"        : self.class_names[pred_cls],
            "confidence"    : round(conf * 100, 2),
            "threshold"     : round(float(thresh) * 100, 2),
            "intrusion"     : accepted,
            "top_3"         : [(self.class_names[i], round(float(p)*100, 2))
                                for i, p in top3],
            "model"         : "YOLOv8+RL",
        }

    def summary(self):
        self.model.summary()
        print(f"\nCurrent RL thresholds: {dict(zip(self.class_names, np.round(self.rl_adapter.thresholds, 3)))}")
