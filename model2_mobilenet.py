"""
models/model2_mobilenet.py
═══════════════════════════════════════════════════════════════════
MODEL 2 — MobileNetV2 Transfer Learning + RL Adaptation
═══════════════════════════════════════════════════════════════════
Architecture:
  • MobileNetV2 pre-trained on ImageNet as feature extractor
  • Custom dense head fine-tuned for farm animal classes
  • Two-phase training:
      Phase 1 — freeze backbone, train only dense head (10 epochs)
      Phase 2 — unfreeze top 30 layers, fine-tune end-to-end (20 epochs)
  • Same RL Policy Adapter as Model 1 for fair comparison

Why MobileNetV2 for edge?
  • Depthwise separable convolutions → 8–9× fewer parameters than VGG
  • Designed for mobile/embedded inference
  • ImageNet pre-training gives strong general visual features
  • Easily exported to TFLite for Raspberry Pi / Jetson Nano
═══════════════════════════════════════════════════════════════════
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from model1_custom_cnn import RLPolicyAdapter   # reuse same RL agent


# ─────────────────────────────────────────────────────────────────
# MOBILENETV2 TRANSFER LEARNING MODEL
# ─────────────────────────────────────────────────────────────────
def build_mobilenet_model(input_shape=(224, 224, 3), num_classes=10):
    """
    Builds MobileNetV2 with a custom classification head.
    Backbone weights are loaded from ImageNet.
    """
    base_model = keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False   # freeze backbone for phase 1

    inputs = keras.Input(shape=input_shape, name="image_input")

    # MobileNetV2 expects pixel values in [-1, 1]
    x = keras.applications.mobilenet_v2.preprocess_input(inputs * 255.0)
    x = base_model(x, training=False)

    # Custom head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu',
                     kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation='relu',
                     kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax',
                           name='predictions')(x)

    model = keras.Model(inputs, outputs, name="MobileNetV2_FarmIntruder")
    return model, base_model


# ─────────────────────────────────────────────────────────────────
# COMBINED WRAPPER
# ─────────────────────────────────────────────────────────────────
class MobileNetV2_RL:
    """
    Full model = MobileNetV2 backbone + custom head + RL policy adapter.
    Two-phase training strategy to avoid catastrophic forgetting.
    """

    def __init__(self, class_names, img_size=(224, 224, 3),
                 rl_lr=1e-3, gamma=0.99):
        self.class_names    = class_names
        self.num_classes    = len(class_names)
        self.img_size       = img_size
        self.model, self.base_model = build_mobilenet_model(img_size, self.num_classes)
        self.rl_adapter     = RLPolicyAdapter(self.num_classes, rl_lr, gamma)
        self.history        = {'accuracy': [], 'val_accuracy': [],
                               'loss': [],     'val_loss': []}
        self.rl_loss_log    = []

    # ── Two-phase transfer learning ──────────────────────────────
    def compile_and_train(self, train_ds, val_ds,
                          phase1_epochs=10, phase2_epochs=20,
                          phase1_lr=1e-3, phase2_lr=1e-5):
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy', patience=6,
                restore_best_weights=True, verbose=1),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5,
                patience=3, min_lr=1e-7, verbose=1),
        ]

        # ── Phase 1: train head only ──────────────────────────
        print("\n[MobileNetV2] Phase 1: Training custom head (backbone frozen)")
        self.model.compile(
            optimizer=keras.optimizers.Adam(phase1_lr),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        h1 = self.model.fit(train_ds, validation_data=val_ds,
                            epochs=phase1_epochs, callbacks=callbacks, verbose=1)
        self._merge_history(h1.history)

        # ── Phase 2: fine-tune top layers ─────────────────────
        print("\n[MobileNetV2] Phase 2: Fine-tuning top 30 layers")
        self.base_model.trainable = True
        # Freeze all but the last 30 layers
        for layer in self.base_model.layers[:-30]:
            layer.trainable = False

        self.model.compile(
            optimizer=keras.optimizers.Adam(phase2_lr),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        h2 = self.model.fit(train_ds, validation_data=val_ds,
                            epochs=phase2_epochs, callbacks=callbacks, verbose=1)
        self._merge_history(h2.history)

        return self.history

    def _merge_history(self, h):
        for k in ['accuracy', 'val_accuracy', 'loss', 'val_loss']:
            self.history[k].extend(h.get(k, []))

    # ── Inference ────────────────────────────────────────────────
    def predict(self, X):
        return self.model.predict(X, verbose=0)

    def predict_with_rl(self, image):
        probs     = self.model.predict(image[np.newaxis], verbose=0)[0]
        action, _ = self.rl_adapter.select_action(probs)
        confidence= float(probs[action])
        return self.class_names[action], confidence, action

    # ── RL online adaptation ─────────────────────────────────────
    def rl_adapt_episode(self, images, true_labels, verbose=False):
        episode_rewards = []
        for img, true_lbl in zip(images, true_labels):
            animal, conf, action = self.predict_with_rl(img)
            reward = 1.0 if action == true_lbl else -1.0
            self.rl_adapter.store_reward(reward)
            episode_rewards.append(reward)

        loss = self.rl_adapter.update()
        self.rl_loss_log.append(loss)

        if verbose:
            print(f"  [RL Episode {self.rl_adapter.total_episodes}] "
                  f"Avg Reward: {np.mean(episode_rewards):.3f}  "
                  f"Policy Loss: {loss:.4f}")
        return episode_rewards

    # ── Detection output ──────────────────────────────────────────
    def detect(self, image_array):
        animal, conf, idx = self.predict_with_rl(image_array)
        probs = self.model.predict(image_array[np.newaxis], verbose=0)[0]
        top3  = sorted(enumerate(probs), key=lambda x: -x[1])[:3]
        return {
            "animal"     : animal,
            "confidence" : round(conf * 100, 2),
            "intrusion"  : conf > 0.5,
            "top_3"      : [(self.class_names[i], round(float(p)*100, 2))
                             for i, p in top3],
            "model"      : "MobileNetV2+RL",
        }

    # ── Export to TFLite (for Raspberry Pi / Jetson) ──────────────
    def export_tflite(self, save_path="mobilenet_farm.tflite", quantize=True):
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        if quantize:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        with open(save_path, 'wb') as f:
            f.write(tflite_model)
        size_mb = len(tflite_model) / (1024**2)
        print(f"[TFLite] Saved to {save_path} ({size_mb:.2f} MB)")
        return save_path

    def summary(self):
        self.model.summary()
