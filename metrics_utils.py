"""
utils/metrics_utils.py
Comprehensive evaluation metrics:
  - Accuracy, Precision, Recall, F1 (per-class & macro)
  - mAP (mean Average Precision)
  - Confusion Matrix
  - Inference latency
  - Edge suitability score
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
    average_precision_score
)
from sklearn.preprocessing import label_binarize


# ─────────────────────────────────────────────
# CORE METRICS
# ─────────────────────────────────────────────
def compute_all_metrics(y_true, y_pred, y_proba, class_names, model_name):
    """
    Compute and return a full metrics dictionary for one model.

    Args:
        y_true   : 1-D array of ground-truth class indices
        y_pred   : 1-D array of predicted class indices
        y_proba  : 2-D array (N x num_classes) of class probabilities
        class_names : list of class label strings
        model_name  : string tag for this model

    Returns:
        dict with all metric values
    """
    n_classes = len(class_names)
    y_bin     = label_binarize(y_true, classes=list(range(n_classes)))

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec  = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1   = f1_score(y_true, y_pred, average='macro', zero_division=0)

    # Per-class AP then mAP
    per_class_ap = {}
    for i, cname in enumerate(class_names):
        if y_bin[:, i].sum() > 0:
            per_class_ap[cname] = average_precision_score(y_bin[:, i], y_proba[:, i])
        else:
            per_class_ap[cname] = 0.0
    mAP = float(np.mean(list(per_class_ap.values())))

    # Per-class F1
    f1_per = f1_score(y_true, y_pred, average=None, zero_division=0)
    per_class_f1 = {class_names[i]: float(f1_per[i]) for i in range(len(f1_per))}

    return {
        "model"         : model_name,
        "accuracy"      : round(acc,  4),
        "precision"     : round(prec, 4),
        "recall"        : round(rec,  4),
        "f1_macro"      : round(f1,   4),
        "mAP"           : round(mAP,  4),
        "per_class_AP"  : {k: round(v, 4) for k, v in per_class_ap.items()},
        "per_class_F1"  : {k: round(v, 4) for k, v in per_class_f1.items()},
    }


# ─────────────────────────────────────────────
# LATENCY BENCHMARK
# ─────────────────────────────────────────────
def measure_inference_latency(predict_fn, X_sample, n_runs=50):
    """
    Measure average inference latency for a single image.

    Args:
        predict_fn : callable that accepts a batch of images
        X_sample   : array to sample from
        n_runs     : number of timed runs

    Returns:
        avg_ms : average latency in milliseconds
        fps    : frames per second
    """
    sample = X_sample[:1]  # single image batch
    times  = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        predict_fn(sample)
        times.append(time.perf_counter() - t0)

    avg_ms = float(np.mean(times)) * 1000
    fps    = 1.0 / float(np.mean(times))
    return round(avg_ms, 3), round(fps, 2)


# ─────────────────────────────────────────────
# EDGE SUITABILITY SCORE
# ─────────────────────────────────────────────
def edge_score(model_size_mb, avg_latency_ms, accuracy):
    """
    Composite edge-suitability score (0-100).
    Balances model size, speed, and accuracy for deployment on
    resource-constrained edge devices (Raspberry Pi / Jetson Nano).
    """
    size_score     = max(0, 100 - model_size_mb * 2)   # penalise large models
    latency_score  = max(0, 100 - avg_latency_ms)       # penalise slow models
    accuracy_score = accuracy * 100
    score = 0.35 * size_score + 0.35 * latency_score + 0.30 * accuracy_score
    return round(score, 2)


# ─────────────────────────────────────────────
# PLOT: CONFUSION MATRIX
# ─────────────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred, class_names, model_name, save_path=None):
    cm   = confusion_matrix(y_true, y_pred)
    norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(norm, annot=True, fmt='.2f', cmap='YlOrRd',
                xticklabels=class_names, yticklabels=class_names,
                linewidths=0.5, ax=ax)
    ax.set_xlabel('Predicted Label', fontsize=13)
    ax.set_ylabel('True Label', fontsize=13)
    ax.set_title(f'Confusion Matrix — {model_name}', fontsize=15, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()


# ─────────────────────────────────────────────
# PLOT: TRAINING HISTORY
# ─────────────────────────────────────────────
def plot_training_history(history, model_name, save_path=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    ax1.plot(history['accuracy'],     label='Train Acc', color='#2563eb', linewidth=2)
    ax1.plot(history['val_accuracy'], label='Val Acc',   color='#16a34a', linewidth=2, linestyle='--')
    ax1.set_title(f'{model_name} — Accuracy', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Accuracy')
    ax1.legend(); ax1.grid(alpha=0.3)

    # Loss
    ax2.plot(history['loss'],     label='Train Loss', color='#dc2626', linewidth=2)
    ax2.plot(history['val_loss'], label='Val Loss',   color='#ea580c', linewidth=2, linestyle='--')
    ax2.set_title(f'{model_name} — Loss', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Loss')
    ax2.legend(); ax2.grid(alpha=0.3)

    plt.suptitle(f'Training Curves — {model_name}', fontsize=15, y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ─────────────────────────────────────────────
# PLOT: COMPARISON BAR CHART
# ─────────────────────────────────────────────
def plot_comparison(metrics_list, save_path=None):
    """
    Side-by-side bar chart comparing all three models across key metrics.
    """
    metric_keys  = ['accuracy', 'precision', 'recall', 'f1_macro', 'mAP']
    metric_labels= ['Accuracy', 'Precision', 'Recall', 'F1 (Macro)', 'mAP']
    model_names  = [m['model'] for m in metrics_list]
    colors       = ['#2563eb', '#16a34a', '#dc2626']

    x      = np.arange(len(metric_keys))
    width  = 0.25
    fig, ax = plt.subplots(figsize=(14, 7))

    for i, (metrics, color) in enumerate(zip(metrics_list, colors)):
        vals = [metrics[k] for k in metric_keys]
        bars = ax.bar(x + i * width, vals, width, label=metrics['model'],
                      color=color, alpha=0.85, edgecolor='white', linewidth=0.8)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=8.5, fontweight='bold')

    ax.set_xticks(x + width)
    ax.set_xticklabels(metric_labels, fontsize=12)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel('Score', fontsize=13)
    ax.set_title('Model Comparison — Farm Animal Intrusion Detection',
                 fontsize=15, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()


# ─────────────────────────────────────────────
# PLOT: PER-CLASS AP HEATMAP
# ─────────────────────────────────────────────
def plot_per_class_ap(metrics_list, save_path=None):
    model_names = [m['model'] for m in metrics_list]
    class_names = list(metrics_list[0]['per_class_AP'].keys())
    data = np.array([
        [m['per_class_AP'].get(c, 0.0) for c in class_names]
        for m in metrics_list
    ])
    fig, ax = plt.subplots(figsize=(14, 5))
    sns.heatmap(data, annot=True, fmt='.3f', cmap='Blues',
                xticklabels=class_names, yticklabels=model_names,
                linewidths=0.5, vmin=0, vmax=1, ax=ax)
    ax.set_title('Per-Class Average Precision (AP) by Model',
                 fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()


# ─────────────────────────────────────────────
# PRINT SUMMARY TABLE
# ─────────────────────────────────────────────
def print_summary_table(metrics_list, latency_data=None):
    rows = []
    for m in metrics_list:
        row = {
            'Model'    : m['model'],
            'Accuracy' : f"{m['accuracy']:.4f}",
            'Precision': f"{m['precision']:.4f}",
            'Recall'   : f"{m['recall']:.4f}",
            'F1 Macro' : f"{m['f1_macro']:.4f}",
            'mAP'      : f"{m['mAP']:.4f}",
        }
        if latency_data and m['model'] in latency_data:
            row['Latency (ms)'] = latency_data[m['model']]['latency_ms']
            row['FPS']          = latency_data[m['model']]['fps']
        rows.append(row)

    df = pd.DataFrame(rows)
    print("\n" + "="*80)
    print("  FINAL RESULTS — AI-BASED FARM ANIMAL INTRUSION PREVENTION SYSTEM")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80 + "\n")
    return df
