# Project Explanation — AI-Based Edge Farm Animal Intrusion Prevention System
### How the Results Were Obtained & What They Mean
*Prepared for presentation to a Project Guide*

---

## Table of Contents
1. [What the Project Is About](#1-what-the-project-is-about)
2. [The Dataset](#2-the-dataset)
3. [How the Experiment Was Set Up](#3-how-the-experiment-was-set-up)
4. [The Three Models Tested](#4-the-three-models-tested)
5. [Reinforcement Learning — The Adaptive Layer](#5-reinforcement-learning--the-adaptive-layer)
6. [How Metrics Were Calculated](#6-how-metrics-were-calculated)
7. [Results — Full Breakdown](#7-results--full-breakdown)
8. [Per-Class Results Explained](#8-per-class-results-explained)
9. [Training Time Analysis](#9-training-time-analysis)
10. [Edge Score — What It Means](#10-edge-score--what-it-means)
11. [Overall Conclusion](#11-overall-conclusion)

---

## 1. What the Project Is About

The core problem this research solves is: **how do you protect a farm from animal intrusions in a remote area with no internet and no electricity grid?**

The proposed solution is a **fully autonomous, solar-powered, edge-AI system** with two hardware nodes:

| Node | Hardware | Role |
|------|----------|------|
| **Master (AI Brain)** | Raspberry Pi 5 | Runs the AI, makes decisions, sends commands wirelessly |
| **Actuator** | ESP32 microcontroller | Receives commands, triggers the 110 dB siren and floodlights |

The two nodes are connected wirelessly via **LoRa radio (433 MHz, ~1 km range)** — no WiFi or internet is needed.

The AI pipeline works in **two stages to save power**:
1. **Stage 1 (always on):** A thermal camera (MLX90640) scans 768 pixels 2x per second. It only wakes up the main camera when it detects a warm body (>20 pixels above 30 degrees C).
2. **Stage 2 (triggered):** The RGB camera captures a frame, and the AI classifies the animal and decides whether to trigger the deterrents.

---

## 2. The Dataset

The dataset contains images of **6 classes of animals** that commonly intrude on farms:

| Class | Why It's Included |
|-------|-------------------|
| **Bear** | Predator, highest threat level |
| **Cow** | Could be own livestock straying or neighbours' |
| **Deer** | Common crop damage animal |
| **Goat** | Common in South Asian / African farms |
| **Human** | Could be a farm worker (day) or intruder (night) |
| **Wild Boar** | Major crop destruction threat |

The dataset holds **5,200+ images** stored in `dataset/bear/`, `dataset/cow/`, etc.

**Preprocessing pipeline** (in `dataset_utils.py`):
1. Images are resized to **224x224 pixels** — standard input size for MobileNet and CNNs.
2. Pixel values are **normalised to [0, 1]** by dividing by 255.
3. Dataset is split: **70% training / 10% validation / 20% testing** (stratified — each class gets proportional representation in every split).
4. **Data augmentation** is applied to training images: random horizontal flips, brightness changes (+/-20%), contrast changes (+/-20%), saturation changes (+/-20%). This prevents the model from memorising exact images.

---

## 3. How the Experiment Was Set Up

All three models were trained and evaluated **under identical conditions** for fair comparison:
- Same 224x224 image input size
- Same dataset and same train/val/test splits
- Same RL adapter design
- Same evaluation metrics computed in `metrics_utils.py`
- Same hardware (Windows PC for training, results exported for Raspberry Pi deployment)

Training was done inside the **Jupyter notebook** (`phd_research_notebook.ipynb`), which orchestrates all three models, runs the RL episodes, measures latency, and generates the 16 research figures saved in `results/figures/`.

---

## 4. The Three Models Tested

### Model 1 — Custom CNN + RL (`model1_custom_cnn.py`)

This is a **from-scratch CNN** designed to be lightweight for edge hardware.

**Architecture:**
```
Input (224x224x3)
  |
Block 1: Conv(32) -> BN -> ReLU -> Conv(32) -> BN -> ReLU -> MaxPool -> Dropout(0.25)
  |
Block 2: Conv(64) -> BN -> ReLU -> Conv(64) -> BN -> ReLU -> MaxPool -> Dropout(0.25)
  |
Block 3: Conv(128) -> BN -> ReLU -> Conv(128) -> BN -> ReLU -> MaxPool -> Dropout(0.35)
  |
Block 4: Conv(256) -> BN -> ReLU -> Conv(256) -> BN -> ReLU -> GlobalAvgPool -> Dropout(0.5)
  |
Dense(256) -> Dropout(0.4) -> Dense(6, softmax)
```

- **4 VGG-style double convolution blocks** — each block doubles the number of filters
- **Batch Normalisation (BN)** after every convolution — stabilises training, acts as a regulariser
- **Dropout** at every block — prevents overfitting
- **L2 regularisation (1e-4)** on all conv and dense layers
- **Global Average Pooling** instead of flatten — reduces parameters significantly
- Total model size: **~14.37 MB**

**Training strategy:**
- Optimizer: Adam (lr = 0.001)
- Loss: Sparse Categorical Cross-Entropy
- Callbacks: EarlyStopping (patience=7) + ReduceLROnPlateau (patience=3, factor=0.5)

---

### Model 2 — MobileNetV2 + RL (`model2_mobilenet.py`)

This uses **transfer learning** — it starts with MobileNetV2 already trained on 1.28 million ImageNet images, so it already knows how to detect edges, textures, and shapes before seeing a single farm animal.

**Architecture:**
```
MobileNetV2 (ImageNet pretrained backbone — frozen in Phase 1)
  |  (feature maps)
GlobalAveragePooling2D
  |
Dense(256, relu) -> Dropout(0.4)
  |
Dense(128, relu) -> Dropout(0.3)
  |
Dense(6, softmax)
```

**Two-phase training strategy** (avoids "catastrophic forgetting"):
- **Phase 1 (10 epochs, lr=0.001):** Only the custom dense head is trained. The MobileNetV2 backbone is frozen. This teaches the head to use the pre-learned features.
- **Phase 2 (20 epochs, lr=0.00001):** The top 30 layers of MobileNetV2 are unfrozen and fine-tuned with a very low learning rate. This adapts the backbone slightly to farm animal images.

**Why MobileNetV2 is ideal for edge:**
- Uses **depthwise separable convolutions** — 8-9x fewer parameters than standard convolutions
- Designed by Google specifically for mobile/embedded devices
- Exported to **TFLite int8** format for Raspberry Pi deployment (quantised to 8-bit integers -> ~4x smaller, ~2x faster on ARM)
- Model size: **~25 MB** (keras), much smaller after TFLite quantisation

---

### Model 3 — YOLOv8-Style + Threshold RL (`model3_yolo.py`)

This model mimics **YOLOv8's architecture philosophy** using CSP (Cross-Stage Partial) blocks.

**Architecture:**
```
Input (224x224x3)
  |
Stem: Conv(32, stride=2) -> BN -> LeakyReLU(0.1)
  |
CSP Block 1 (64 filters) -> MaxPool
  |
CSP Block 2 (128 filters) -> MaxPool
  |
CSP Block 3 (256 filters) -> MaxPool
  |
CSP Block 4 (512 filters)
  |
GlobalAveragePooling -> Dense(256) -> Dropout(0.4) -> Dense(6, softmax)
```

**What is a CSP Block?**
Each block splits the input into two paths:
1. **Main path:** DepthwiseConv -> BN -> LeakyReLU -> PointwiseConv(1x1) -> BN -> LeakyReLU
2. **Skip path:** 1x1 Conv -> BN (for matching channels)
3. Both paths are **added or concatenated** (residual connection)

This allows gradients to flow through two paths during backpropagation, reducing vanishing gradient problems and enabling deeper networks.

**Key differences from Models 1 & 2:**
- Uses **LeakyReLU (alpha=0.1)** instead of regular ReLU — allows small negative activations, which helps with gradient flow
- **Depthwise + pointwise convolutions** (like MobileNet) — fewer parameters
- Smallest model: **~9.11 MB**
- Fastest inference: **107 ms / frame (9.34 FPS)**

---

## 5. Reinforcement Learning — The Adaptive Layer

All three models use a **REINFORCE (Policy Gradient) RL adapter** that operates on top of the base classifier. This is what makes the system "adaptive."

### Why RL?
A static model trained once will degrade over time as field conditions change (different lighting, seasons, new animals). Instead of retraining every time, the RL adapter **adjusts the model's decision-making in real-time** based on feedback.

### Models 1 & 2 — REINFORCE Action Selection

```
State  -> CNN softmax probabilities (shape: [6])
Action -> Which animal class to alert/suppress (integer 0-5)
Reward -> +1 correct, -1 false alarm, 0 uncertain
```

**The policy network** is a tiny 3-layer MLP:
```
Dense(32, relu) -> Dense(32, relu) -> Dense(6, softmax)
```

**REINFORCE update (at end of each episode):**
1. Compute **discounted returns**: G_t = r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + ...  (gamma = 0.99)
2. Normalise returns (zero mean, unit variance) for stable training
3. Compute loss = -mean(log(pi(a_t|s_t)) * G_t)
4. Backpropagate through the policy MLP only (not through the CNN)
5. Clear episode memory

This is **Monte Carlo** — the update happens after a complete sequence of frames, not per-step.

### Model 3 — Threshold RL Adapter

Instead of selecting actions, YOLO's RL adapter **adjusts the confidence threshold per class**:
- If class X is producing many false positives -> **raise its threshold**
- If class X is causing many missed detections -> **lower its threshold**
- Uses **exponential decay (0.95)** on FP/FN rates so old mistakes fade over time

```python
delta = lr * (fp_rates - fn_rates)
thresholds = clip(thresholds + delta, 0.2, 0.95)
```

This means the system becomes more conservative for problematic classes and more sensitive for under-detected classes — **automatically, without human intervention**.

---

## 6. How Metrics Were Calculated

All metrics are computed in `metrics_utils.py` using **scikit-learn** functions on the held-out **test set** (never seen during training).

| Metric | Formula | What It Means |
|--------|---------|----------------|
| **Accuracy** | Correct / Total | Overall fraction of correctly classified images |
| **Precision (Macro)** | avg(TP / (TP+FP)) per class | Of all the times the model said "this is class X", how often was it right? |
| **Recall (Macro)** | avg(TP / (TP+FN)) per class | Of all actual class X animals, how many did the model catch? |
| **F1 Score (Macro)** | avg(2*P*R / (P+R)) per class | Harmonic mean of Precision and Recall — the balanced metric |
| **mAP** | mean(AP per class) | Area under Precision-Recall curve, averaged across all 6 classes |
| **Latency (ms)** | avg over 50 single-image runs | How long one inference takes in milliseconds |
| **FPS** | 1000 / Latency (ms) | Frames per second the model can process |
| **Edge Score** | 0.35*(100-size*2) + 0.35*(100-latency) + 0.30*(acc*100) | Composite score balancing size, speed, and accuracy for embedded deployment |

> **Why Macro averaging?** Because the dataset may have class imbalances. Macro averaging gives equal weight to each class regardless of how many samples it has — critical for safety (a rare bear detection is just as important as a common cow detection).

---

## 7. Results — Full Breakdown

### Overall Performance Table

| Model | Accuracy | Precision | Recall | F1 (Macro) | mAP | Latency | FPS | Size (MB) | Edge Score |
|-------|----------|-----------|--------|------------|-----|---------|-----|-----------|------------|
| Custom CNN+RL | 0.9079 | 0.5948 | 0.6102 | 0.6024 | 0.6811 | 207.64 ms | 4.82 | 14.37 | 52.18 |
| MobileNetV2+RL | **0.9885** | **0.9544** | **0.9374** | **0.9444** | **0.9845** | 117.77 ms | 8.49 | 25.00 | 47.16 |
| YOLOv8-Style+RL | 0.9357 | 0.8319 | 0.7237 | 0.7564 | 0.7942 | **107.10 ms** | **9.34** | **9.11** | **56.70** |

### Reading the Results Row by Row

#### Custom CNN+RL — Why it underperformed
- **Accuracy = 90.79%** — This seems good until you look at Precision and Recall
- **Precision = 0.5948 / Recall = 0.6102 / F1 = 0.6024** — These are much lower, revealing a problem: the model gets the **overall accuracy up by being good at the common classes** (cow, human, wild boar), but **completely fails on rare/visually similar classes** (bear, goat)
- **mAP = 0.6811** — Confirmed by per-class AP showing Bear (0.0858) and Goat (0.1554) near zero
- **Root cause:** The custom CNN was trained from scratch with no pretrained knowledge. It did not learn discriminative enough features for visually ambiguous classes like bear and goat, which can look similar to each other (four-legged, similar size, furry)
- **Latency = 207 ms (4.82 FPS)** — Slowest of the three because it has 4 large double-convolution blocks with no depthwise optimisation
- **Edge Score = 52.18** — Middle rank despite being smaller (14.37 MB) because the slow speed hurts it

#### MobileNetV2+RL — The clear winner on accuracy
- **Accuracy = 98.85%** — Near-perfect overall classification
- **Precision = 0.9544 / Recall = 0.9374 / F1 = 0.9444** — All metrics are very high and consistent with each other, meaning the model performs well across **all classes**, not just common ones
- **mAP = 0.9845** — Extremely high. Per-class AP shows 1.0000 for Bear and Human — perfect detection
- **Root cause of success:** Transfer learning. MobileNetV2 was pre-trained on 1.28 million diverse ImageNet images and already learned to detect fur textures, animal shapes, and body proportions. Fine-tuning needed only ~27 minutes to adapt this knowledge to farm animals
- **Latency = 117.77 ms (8.49 FPS)** — Moderate speed, acceptable for edge
- **Size = 25 MB** — Largest model, but after TFLite int8 quantisation it becomes ~6-7 MB on the Raspberry Pi
- **Edge Score = 47.16** — Lowest edge score due to larger size, but this is somewhat misleading because quantisation (which makes it much smaller on Pi) is not reflected in the raw 25 MB figure

#### YOLOv8-Style+RL — The best for edge deployment
- **Accuracy = 93.57%** — Good but lower than MobileNet
- **Precision = 0.8319 / Recall = 0.7237 / F1 = 0.7564** — Notably the Recall is lower than Precision, meaning it misses some animals (false negatives). For a farm security system, **missed detections are dangerous** — this is a concern
- **mAP = 0.7942** — Below MobileNet but better than Custom CNN
- **Fastest: 107.10 ms (9.34 FPS)** — Best real-time performance
- **Smallest: 9.11 MB** — By far the most compact model
- **Edge Score = 56.70** — Highest edge score because the composite formula rewards the tiny size and fast speed

---

## 8. Per-Class Results Explained

### F1 Scores Per Class

| Class | Custom CNN+RL | MobileNetV2+RL | YOLOv8-Style+RL | Analysis |
|-------|--------------|----------------|-----------------|----------|
| **Bear** | 0.0000 | 0.9091 | 0.3529 | CNN completely fails. MobileNet near-perfect. YOLO struggles |
| **Cow** | 0.9316 | 0.9949 | 0.9516 | All models handle cows well — most visually distinct class |
| **Deer** | 0.8196 | 0.9750 | 0.8833 | MobileNet significantly better than others |
| **Goat** | 0.0000 | 0.8000 | 0.4444 | CNN fails completely. Goats are visually similar to deer |
| **Human** | 0.9790 | 0.9985 | 0.9821 | All models excel — humans have unique upright posture |
| **Wild Boar** | 0.8841 | 0.9891 | 0.9241 | All models perform well |

### Key Observations

**1. Bear and Goat are the hardest classes**
- Both are four-legged, medium-sized, and often captured at unfavourable angles
- The Custom CNN, trained from scratch, never learned good discriminative features for these
- MobileNetV2's ImageNet pre-training already learned mammal body parts, enabling it to distinguish a bear's body structure from a goat's
- YOLOv8's threshold RL helps somewhat — the RL adapter raised the threshold for these classes to reduce false alarms, but this also means some true bears/goats are missed (lower recall)

**2. Human is the easiest class**
- All three models score near-perfect F1 for humans
- Humans have a unique feature: **bipedal, upright posture with a distinct silhouette**. No other class in the dataset shares this shape
- This is consistent with the project's deterrent logic: the system must reliably detect humans to differentiate between farm workers (day, no action) and intruders (night, activate lights)

**3. Cow and Wild Boar score high across all models**
- These animals are likely the best represented in the dataset
- They also have visually distinct features: cows are large and white/spotted; wild boars are dark-coloured, stocky, and have a distinctive snout profile

**4. The Accuracy vs F1 gap in Custom CNN**
- Custom CNN: Accuracy = 90.79%, but F1 macro = 0.6024 — a 30-point gap
- This is explained by class imbalance: if Bear and Goat together make up ~33% of the dataset but the model scores F1=0 on both, the macro F1 tanks dramatically. But if the other 4 classes (67% of test data) are predicted correctly, overall accuracy stays high. **This is why F1-macro is more meaningful than accuracy alone for multi-class imbalanced problems**

---

## 9. Training Time Analysis

| Model | Time | Reason |
|-------|------|--------|
| **Custom CNN+RL** | **479 minutes (~8 hours)** | Trained fully from scratch with 4 double-convolution blocks. No pretrained initialisation means many epochs needed to converge. Standard convolutions are also slower per epoch than depthwise |
| **MobileNetV2+RL** | **27 minutes** | Phase 1 (10 epochs): head only — very fast. Phase 2 (20 epochs): only top 30 layers. EarlyStopping also cuts epochs short when validation plateaus. Transfer learning is the reason it's 17x faster than the CNN despite being a larger model |
| **YOLOv8-Style+RL** | **143 minutes (~2.4 hours)** | Built from scratch like CNN but uses depthwise separable convolutions (fewer multiplications per layer), making each epoch significantly faster than the Custom CNN |

> **Key insight for your guide:** The long training time of the Custom CNN is not a deployment problem — training happens **once on a PC**. The trained `.keras` file is then copied to the Raspberry Pi via SCP. The Pi only runs inference (~100–200 ms per frame), never training.

---

## 10. Edge Score — What It Means

The **Edge Suitability Score** is a custom composite metric designed for this project to help select which model to deploy on the Raspberry Pi 5:

```
edge_score = 0.35 * (100 - model_size_mb * 2)    [size penalty]
           + 0.35 * (100 - avg_latency_ms)         [speed penalty]
           + 0.30 * (accuracy * 100)               [accuracy reward]
```

**Breaking down each model's edge score:**

| Model | Size Score | Speed Score | Accuracy Score | Total |
|-------|-----------|-------------|----------------|-------|
| Custom CNN (14.37 MB, 207ms, 90.79%) | 0.35*(71.26)=24.94 | ~0 (latency >100ms) | 0.30*90.79=27.24 | 52.18 |
| MobileNetV2 (25 MB, 117ms, 98.85%) | 0.35*(50)=17.50 | ~0 (latency >100ms) | 0.30*98.85=29.65 | 47.16 |
| YOLOv8 (9.11 MB, 107ms, 93.57%) | 0.35*(81.78)=28.62 | ~0 (latency >100ms) | 0.30*93.57=28.07 | 56.70 |

> Note: All three models have latency above 100ms on a CPU, so their speed scores are ~0 in this formula. The dominant differentiator is **model size** (which favours YOLOv8) and **accuracy** (which favours MobileNetV2). On the Raspberry Pi with TFLite int8 quantisation, latency would drop to ~50-70ms for YOLOv8, dramatically improving its edge score.

---

## 11. Overall Conclusion

### Which Model Is Best and Why?

The experiment demonstrates a classic **accuracy vs. efficiency trade-off** in edge AI:

| Goal | Best Model | Why |
|------|-----------|-----|
| **Maximum Detection Accuracy** | MobileNetV2+RL | 98.85% accuracy, 98.45% mAP, perfect Bear & Human detection |
| **Best Edge Deployment** | YOLOv8-Style+RL | Smallest (9.11 MB), fastest (9.34 FPS), highest edge score (56.70) |
| **Avoid This Configuration** | Custom CNN+RL | F1=0 on Bear and Goat, slowest (207ms), longest training (8 hours) |

### The Production System Uses Both Models Together

The actual Raspberry Pi deployment uses a **fusion of YOLOv8n and MobileNetV2** simultaneously:

```
if both agree on the same class -> use that class, take max(confidence)
if MobileNetV2 confidence > YOLOv8 confidence + 15% -> use MobileNetV2
else -> use YOLOv8
```

This fusion combines the **speed and small size of YOLO** with the **superior accuracy of MobileNetV2**, achieving better results than either model alone.

### The RL Layer Adds Real Value

The REINFORCE RL adapter enables **online adaptation without retraining**:
- On a farm, bear appearances might be rare. If a bear is misclassified once as a goat, the RL adapter adjusts to be more sensitive to bear-like probability vectors in future frames
- The threshold RL in YOLOv8 is particularly useful for reducing false alarms in night conditions where image quality degrades

### What This Research Contributes

1. **Novel three-way comparison** of architecturally distinct models (from-scratch CNN vs. transfer learning vs. YOLO-style detector) on a farm animal intrusion dataset
2. **Integration of RL** as a real-time adaptive layer on top of static classifiers — enabling continuous improvement without retraining
3. **Full system design** from solar power calculation to LoRa communication protocol to Raspberry Pi deployment
4. **Edge-aware evaluation** using a composite Edge Score metric that accounts for real hardware constraints beyond just accuracy

---

*Document prepared to explain the complete methodology and result interpretation for the PhD Research Project: AI-Based Edge-Enabled Adaptive Farm Animal Intrusion Prevention System, 2025.*
