"""
models/model1_custom_cnn.py
═══════════════════════════════════════════════════════════════════
MODEL 1 — Custom CNN + Reinforcement Learning Adaptation
═══════════════════════════════════════════════════════════════════
Architecture:
  • 4 convolutional blocks with batch normalisation and dropout
  • Global average pooling → dense head
  • RL adapter: a lightweight policy network that adjusts confidence
    thresholds and class weights online based on reward signals
    (correct / incorrect detections from the field).

RL Strategy: Policy Gradient (REINFORCE)
  State  : model's softmax probability vector for the current frame
  Action : which animal class to alert / suppress
  Reward : +1 correct detection, -1 false alarm, 0 no detection
  Update : gradient ascent on log-probability of chosen action
           weighted by cumulative reward
═══════════════════════════════════════════════════════════════════
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers


# ─────────────────────────────────────────────────────────────────
# CUSTOM CNN ARCHITECTURE
# ─────────────────────────────────────────────────────────────────
def build_custom_cnn(input_shape=(224, 224, 3), num_classes=10):
    """
    Builds the custom CNN backbone.

    Block design:
        Conv2D(3x3) → BatchNorm → ReLU → Conv2D(3x3) → BatchNorm → ReLU
        → MaxPool(2x2) → Dropout(0.25)

    Inspired by VGG-style double convolutions but far smaller
    to run on edge hardware (Raspberry Pi / Jetson Nano).
    """
    inputs = keras.Input(shape=input_shape, name="image_input")

    # ── Block 1 ──────────────────────────────
    x = layers.Conv2D(32, (3, 3), padding='same',
                      kernel_regularizer=regularizers.l2(1e-4),
                      name='conv1a')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(32, (3, 3), padding='same',
                      kernel_regularizer=regularizers.l2(1e-4),
                      name='conv1b')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    # ── Block 2 ──────────────────────────────
    x = layers.Conv2D(64, (3, 3), padding='same',
                      kernel_regularizer=regularizers.l2(1e-4),
                      name='conv2a')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(64, (3, 3), padding='same',
                      kernel_regularizer=regularizers.l2(1e-4),
                      name='conv2b')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    # ── Block 3 ──────────────────────────────
    x = layers.Conv2D(128, (3, 3), padding='same',
                      kernel_regularizer=regularizers.l2(1e-4),
                      name='conv3a')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(128, (3, 3), padding='same',
                      kernel_regularizer=regularizers.l2(1e-4),
                      name='conv3b')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.35)(x)

    # ── Block 4 ──────────────────────────────
    x = layers.Conv2D(256, (3, 3), padding='same',
                      kernel_regularizer=regularizers.l2(1e-4),
                      name='conv4a')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(256, (3, 3), padding='same',
                      kernel_regularizer=regularizers.l2(1e-4),
                      name='conv4b')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)

    # ── Dense Head ───────────────────────────
    x = layers.Dense(256, activation='relu',
                     kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation='softmax',
                           name='predictions')(x)

    model = keras.Model(inputs, outputs, name="CustomCNN_FarmIntruder")
    return model


# ─────────────────────────────────────────────────────────────────
# RL POLICY NETWORK (REINFORCE / Policy Gradient)
# ─────────────────────────────────────────────────────────────────
class RLPolicyAdapter:
    """
    Lightweight policy network that sits on top of the CNN.

    State  : softmax probability vector from the CNN  (shape: [num_classes])
    Action : chosen alert class index                 (int)
    Reward : +1 correct, -1 false alarm, 0 uncertain

    The policy maps state → action probabilities using a small MLP.
    Parameters are updated via REINFORCE after each episode (sequence
    of frames from the edge camera).

    This enables the system to ADAPT online without retraining the
    full CNN — crucial for edge deployment.
    """

    def __init__(self, num_classes=10, lr=1e-3, gamma=0.99):
        self.num_classes   = num_classes
        self.gamma         = gamma          # discount factor
        self.lr            = lr
        self.policy_net    = self._build_policy()
        self.optimizer     = tf.keras.optimizers.Adam(lr)

        # Episode memory
        self.states, self.actions, self.rewards = [], [], []
        self.total_episodes = 0
        self.cumulative_reward = 0.0

    def _build_policy(self):
        """Small MLP: state (probs) → action logits."""
        model = keras.Sequential([
            layers.Dense(32, activation='relu',
                         input_shape=(self.num_classes,)),
            layers.Dense(32, activation='relu'),
            layers.Dense(self.num_classes, activation='softmax'),
        ], name="RL_Policy")
        return model

    def select_action(self, cnn_probs):
        """
        Given CNN's softmax probabilities, the policy chooses an action.
        Uses ε-greedy exploration during early training, then pure policy.
        """
        cnn_probs = np.array(cnn_probs).reshape(1, -1).astype(np.float32)
        policy_probs = self.policy_net(cnn_probs).numpy()[0]
        action = np.random.choice(self.num_classes, p=policy_probs)
        self.states.append(cnn_probs[0])
        self.actions.append(action)
        return action, policy_probs

    def store_reward(self, reward):
        self.rewards.append(reward)
        self.cumulative_reward += reward

    def update(self):
        """
        Perform a REINFORCE policy gradient update at end of episode.
        Returns the policy loss for logging.
        """
        if not self.rewards:
            return 0.0

        # Compute discounted returns
        returns = []
        G = 0.0
        for r in reversed(self.rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = np.array(returns, dtype=np.float32)

        # Normalise returns for stable training
        if returns.std() > 1e-8:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        states  = tf.constant(np.array(self.states),  dtype=tf.float32)
        actions = tf.constant(np.array(self.actions), dtype=tf.int32)
        rets    = tf.constant(returns,                 dtype=tf.float32)

        with tf.GradientTape() as tape:
            probs = self.policy_net(states, training=True)
            # Build indices [[0, a0], [1, a1], ...] to gather each action's prob
            batch_idx = tf.range(tf.shape(actions)[0], dtype=tf.int32)
            gather_idx = tf.stack([batch_idx, actions], axis=1)
            chosen_probs = tf.gather_nd(probs, gather_idx)          # shape [N]
            log_probs = tf.math.log(chosen_probs + 1e-8)            # shape [N]
            loss = -tf.reduce_mean(log_probs * rets)                 # REINFORCE

        grads = tape.gradient(loss, self.policy_net.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grads, self.policy_net.trainable_variables))

        # Clear episode memory
        self.states, self.actions, self.rewards = [], [], []
        self.total_episodes += 1
        return float(loss.numpy())


# ─────────────────────────────────────────────────────────────────
# COMBINED MODEL WRAPPER
# ─────────────────────────────────────────────────────────────────
class CustomCNN_RL:
    """
    Full model = Custom CNN backbone + RL policy adapter.
    Provides a unified interface: train, predict, adapt.
    """

    def __init__(self, class_names, img_size=(224, 224, 3),
                 rl_lr=1e-3, gamma=0.99):
        self.class_names    = class_names
        self.num_classes    = len(class_names)
        self.img_size       = img_size
        self.cnn            = build_custom_cnn(img_size, self.num_classes)
        self.rl_adapter     = RLPolicyAdapter(self.num_classes, rl_lr, gamma)
        self.history        = None
        self.rl_loss_log    = []

    # ── Train CNN backbone ──────────────────────────────────────
    def compile_and_train(self, train_ds, val_ds, epochs=30, lr=1e-3):
        self.cnn.compile(
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

        hist = self.cnn.fit(
            train_ds, validation_data=val_ds,
            epochs=epochs, callbacks=callbacks, verbose=1)
        self.history = hist.history
        return self.history

    # ── Inference: CNN + RL action ──────────────────────────────
    def predict(self, X):
        """Pure CNN prediction (no RL, for evaluation)."""
        return self.cnn.predict(X, verbose=0)

    def predict_with_rl(self, image):
        """
        Full pipeline: CNN → softmax → RL action.
        Returns (animal_name, confidence, rl_action_idx).
        """
        probs      = self.cnn.predict(image[np.newaxis], verbose=0)[0]
        action, _  = self.rl_adapter.select_action(probs)
        confidence = float(probs[action])
        return self.class_names[action], confidence, action

    # ── RL online adaptation ────────────────────────────────────
    def rl_adapt_episode(self, images, true_labels, verbose=False):
        """
        Run one RL episode over a batch of frames.
        After each frame the system receives a reward based on
        whether the RL action matched the true label.
        """
        episode_rewards = []
        for img, true_lbl in zip(images, true_labels):
            animal, conf, action = self.predict_with_rl(img)
            reward = 1.0 if action == true_lbl else -1.0
            self.rl_adapter.store_reward(reward)
            episode_rewards.append(reward)

        loss = self.rl_adapter.update()
        self.rl_loss_log.append(loss)

        if verbose:
            avg_r = np.mean(episode_rewards)
            print(f"  [RL Episode {self.rl_adapter.total_episodes}] "
                  f"Avg Reward: {avg_r:.3f}  Policy Loss: {loss:.4f}")
        return episode_rewards

    # ── Human-readable prediction ───────────────────────────────
    def detect(self, image_array):
        """
        Top-level detection call for edge deployment.
        Returns a structured result dict.
        """
        animal, conf, idx = self.predict_with_rl(image_array)
        probs = self.cnn.predict(image_array[np.newaxis], verbose=0)[0]
        top3  = sorted(enumerate(probs), key=lambda x: -x[1])[:3]
        return {
            "animal"     : animal,
            "confidence" : round(conf * 100, 2),
            "intrusion"  : conf > 0.5,
            "top_3"      : [(self.class_names[i], round(float(p)*100, 2))
                             for i, p in top3],
            "model"      : "CustomCNN+RL",
        }

    def summary(self):
        self.cnn.summary()
        print("\nRL Policy Network:")
        self.rl_adapter.policy_net.summary()
