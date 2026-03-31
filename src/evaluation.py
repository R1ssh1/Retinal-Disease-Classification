"""
Evaluation metrics and visualizations for multi-label retinal disease classification.
"""
import os
import json
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    hamming_loss,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)

# ODIR-5K disease labels
DISEASE_LABELS = ["N", "D", "G", "C", "A", "H", "M", "O"]


def _json_safe(obj):
    """Replace NaN/Inf with None so metrics.json is valid JSON."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        x = float(obj)
        if math.isnan(x) or math.isinf(x):
            return None
        return x
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


def binarize_predictions(y_pred, threshold=0.5):
    """Convert sigmoid outputs to binary 0/1. threshold can be scalar or (n_classes,) per-class."""
    y_pred = np.array(y_pred, dtype=np.float64)
    thr = np.asarray(threshold, dtype=np.float64)
    if thr.ndim == 0:
        return (y_pred >= float(thr)).astype(np.float32)
    return (y_pred >= thr.reshape(1, -1)).astype(np.float32)


def tune_per_class_thresholds_f1(y_true, y_pred_proba, n_steps=37):
    """
    Pick threshold per class that maximizes binary F1 on **validation** labels (not test).
    y_true, y_pred_proba: (n, n_classes).
    """
    y_true = np.asarray(y_true)
    y_pred_proba = np.asarray(y_pred_proba)
    n_classes = y_true.shape[1]
    y_bin = (y_true >= 0.5).astype(np.int32)
    thresholds = np.zeros(n_classes, dtype=np.float64)
    for j in range(n_classes):
        best_t, best_f1 = 0.5, -1.0
        for t in np.linspace(0.05, 0.95, n_steps):
            pred_c = (y_pred_proba[:, j] >= t).astype(np.int32)
            f1 = f1_score(y_bin[:, j], pred_c, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
        thresholds[j] = best_t
    return thresholds


def predict_with_tta(model, X, batch_size=16, verbose=0):
    """
    Average sigmoid outputs over a few deterministic flips (often improves robustness / reported metrics).
    """
    X = np.asarray(X, dtype=np.float32)
    preds = [model.predict(X, batch_size=batch_size, verbose=verbose)]
    preds.append(model.predict(np.flip(X, axis=2), batch_size=batch_size, verbose=verbose))
    preds.append(model.predict(np.flip(X, axis=1), batch_size=batch_size, verbose=verbose))
    return np.mean(np.stack(preds, axis=0), axis=0)


def compute_all_metrics(y_true, y_pred_proba, threshold=0.5):
    """
    Compute comprehensive metrics for multi-label classification.
    y_true, y_pred_proba: (n_samples, n_classes). y_true can be binary or float (binarized at 0.5).
    Returns dict of metric names -> values.
    """
    y_true = np.array(y_true)
    y_true_bin = (y_true >= 0.5).astype(np.int32) if y_true.dtype in (np.float32, np.float64) else y_true
    y_pred = binarize_predictions(y_pred_proba, threshold)
    n_classes = y_true.shape[1]

    metrics = {}

    # Overall metrics (sample-averaged or label-averaged); use binarized y_true for sklearn
    metrics["accuracy"] = float(accuracy_score(y_true_bin, y_pred))
    metrics["precision_macro"] = float(precision_score(y_true_bin, y_pred, average="macro", zero_division=0))
    metrics["precision_micro"] = float(precision_score(y_true_bin, y_pred, average="micro", zero_division=0))
    metrics["recall_macro"] = float(recall_score(y_true_bin, y_pred, average="macro", zero_division=0))
    metrics["recall_micro"] = float(recall_score(y_true_bin, y_pred, average="micro", zero_division=0))
    metrics["f1_macro"] = float(f1_score(y_true_bin, y_pred, average="macro", zero_division=0))
    metrics["f1_micro"] = float(f1_score(y_true_bin, y_pred, average="micro", zero_division=0))
    metrics["hamming_loss"] = float(hamming_loss(y_true_bin, y_pred))
    # Subset accuracy: fraction of samples where all labels match
    subset_correct = np.all(y_true_bin == y_pred, axis=1)
    metrics["subset_accuracy"] = float(np.mean(subset_correct))

    # AUC (multi-label) — use original y_true (can be float)
    try:
        metrics["roc_auc_macro"] = float(roc_auc_score(y_true_bin, y_pred_proba, average="macro", multi_class="ovr"))
    except Exception:
        metrics["roc_auc_macro"] = 0.0
    try:
        metrics["roc_auc_micro"] = float(roc_auc_score(y_true_bin, y_pred_proba, average="micro", multi_class="ovr"))
    except Exception:
        metrics["roc_auc_micro"] = 0.0
    try:
        metrics["average_precision_macro"] = float(average_precision_score(y_true_bin, y_pred_proba, average="macro"))
    except Exception:
        metrics["average_precision_macro"] = 0.0
    try:
        metrics["average_precision_micro"] = float(average_precision_score(y_true_bin, y_pred_proba, average="micro"))
    except Exception:
        metrics["average_precision_micro"] = 0.0

    # Per-class metrics (for reporting)
    metrics["precision_per_class"] = precision_score(y_true_bin, y_pred, average=None, zero_division=0).tolist()
    metrics["recall_per_class"] = recall_score(y_true_bin, y_pred, average=None, zero_division=0).tolist()
    metrics["f1_per_class"] = f1_score(y_true_bin, y_pred, average=None, zero_division=0).tolist()
    return metrics


def plot_training_history(history, save_dir, prefix="training"):
    """Plot loss and metrics over epochs (supports merged history dict)."""
    os.makedirs(save_dir, exist_ok=True)
    if not hasattr(history, "history"):
        return
    h = history.history

    # Loss
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    if "loss" in h:
        ax.plot(h["loss"], label="Train Loss")
    if "val_loss" in h:
        ax.plot(h["val_loss"], label="Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.set_title("Training and Validation Loss")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{prefix}_loss.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Accuracy
    if "accuracy" in h or "val_accuracy" in h:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        if "accuracy" in h:
            ax.plot(h["accuracy"], label="Train Accuracy")
        if "val_accuracy" in h:
            ax.plot(h["val_accuracy"], label="Val Accuracy")
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.set_title("Accuracy")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{prefix}_accuracy.png"), dpi=150, bbox_inches="tight")
        plt.close()

    # AUC
    if "auc" in h or "val_auc" in h:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        if "auc" in h:
            ax.plot(h["auc"], label="Train AUC")
        if "val_auc" in h:
            ax.plot(h["val_auc"], label="Val AUC")
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.set_title("AUC")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{prefix}_auc.png"), dpi=150, bbox_inches="tight")
        plt.close()


def plot_confusion_matrices(y_true, y_pred, save_dir, labels=DISEASE_LABELS):
    """Per-class binary confusion matrices (one per disease)."""
    os.makedirs(save_dir, exist_ok=True)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    n_classes = y_true.shape[1]
    if len(labels) != n_classes:
        labels = [str(i) for i in range(n_classes)]

    for i, name in enumerate(labels):
        try:
            cm = confusion_matrix(y_true[:, i], y_pred[:, i], labels=[0, 1])
            if cm.shape != (2, 2):
                cm = np.zeros((2, 2))
                for a in [0, 1]:
                    for b in [0, 1]:
                        cm[a, b] = np.sum((y_true[:, i] == a) & (y_pred[:, i] == b))
        except Exception:
            cm = np.zeros((2, 2))
        fig, ax = plt.subplots(1, 1, figsize=(4, 3))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1])
        ax.set_ylabel("True")
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Neg", "Pos"])
        ax.set_yticklabels(["Neg", "Pos"])
        ax.set_xlabel("Predicted")
        for r in range(2):
            for c in range(2):
                ax.text(c, r, str(int(cm[r, c])), ha="center", va="center", color="black")
        ax.set_title(f"Confusion Matrix — {name}")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"confusion_matrix_{name}.png"), dpi=150, bbox_inches="tight")
        plt.close()


def plot_roc_curves(y_true, y_pred_proba, save_dir, labels=DISEASE_LABELS):
    """ROC curve per class."""
    os.makedirs(save_dir, exist_ok=True)
    y_true = np.array(y_true)
    y_pred_proba = np.array(y_pred_proba)
    n_classes = y_true.shape[1]
    if len(labels) != n_classes:
        labels = [str(i) for i in range(n_classes)]

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    for i in range(n_classes):
        try:
            fpr, tpr, _ = roc_curve(y_true[:, i], y_pred_proba[:, i])
            auc_val = roc_auc_score(y_true[:, i], y_pred_proba[:, i])
            ax.plot(fpr, tpr, label=f"{labels[i]} (AUC={auc_val:.3f})", lw=2)
        except Exception:
            pass
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves (per class)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "roc_curves.png"), dpi=150, bbox_inches="tight")
    plt.close()


def plot_pr_curves(y_true, y_pred_proba, save_dir, labels=DISEASE_LABELS):
    """Precision-Recall curve per class."""
    os.makedirs(save_dir, exist_ok=True)
    y_true = np.array(y_true)
    y_pred_proba = np.array(y_pred_proba)
    n_classes = y_true.shape[1]
    if len(labels) != n_classes:
        labels = [str(i) for i in range(n_classes)]

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    for i in range(n_classes):
        try:
            prec, rec, _ = precision_recall_curve(y_true[:, i], y_pred_proba[:, i])
            ap = average_precision_score(y_true[:, i], y_pred_proba[:, i])
            ax.plot(rec, prec, label=f"{labels[i]} (AP={ap:.3f})", lw=2)
        except Exception:
            pass
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves (per class)")
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "pr_curves.png"), dpi=150, bbox_inches="tight")
    plt.close()


def plot_metrics_summary(metrics_dict, save_dir):
    """Bar charts for macro metrics and per-class F1/Precision/Recall."""
    os.makedirs(save_dir, exist_ok=True)

    # Macro metrics
    macro_keys = ["precision_macro", "recall_macro", "f1_macro", "roc_auc_macro", "average_precision_macro"]
    names = ["Precision", "Recall", "F1", "ROC AUC", "Avg Precision"]
    values = [metrics_dict.get(k, 0) for k in macro_keys]
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    bars = ax.bar(names, values, color=["#2ecc71", "#3498db", "#9b59b6", "#e74c3c", "#f39c12"])
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title("Overall Metrics (Macro)")
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02, f"{v:.3f}", ha="center", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "metrics_summary.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Per-class F1, Precision, Recall
    if "f1_per_class" in metrics_dict:
        f1_list = np.atleast_1d(metrics_dict["f1_per_class"]).tolist()
        prec_list = np.atleast_1d(metrics_dict.get("precision_per_class", [])).tolist()
        rec_list = np.atleast_1d(metrics_dict.get("recall_per_class", [])).tolist()
        n = len(f1_list)
        prec_list = (prec_list + [0] * n)[:n]
        rec_list = (rec_list + [0] * n)[:n]
        labels = DISEASE_LABELS[:n] if n <= len(DISEASE_LABELS) else [str(i) for i in range(n)]
        x = np.arange(len(labels))
        w = 0.25
        fig, ax = plt.subplots(1, 1, figsize=(12, 5))
        ax.bar(x - w, prec_list, w, label="Precision", color="#3498db")
        ax.bar(x, rec_list, w, label="Recall", color="#2ecc71")
        ax.bar(x + w, f1_list, w, label="F1", color="#9b59b6")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.set_title("Per-Class Precision, Recall, F1")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "per_class_metrics.png"), dpi=150, bbox_inches="tight")
        plt.close()


def plot_feature_importance(y_pred_proba, save_dir, y_true=None, labels=DISEASE_LABELS):
    """
    Feature importance visualization: mean predicted probability per class (model confidence by disease).
    Optional: overlay ground-truth prevalence (mean y_true per class).
    """
    os.makedirs(save_dir, exist_ok=True)
    y_pred_proba = np.array(y_pred_proba)
    n_classes = y_pred_proba.shape[1]
    if len(labels) != n_classes:
        labels = [str(i) for i in range(n_classes)]
    mean_pred = np.mean(y_pred_proba, axis=0)
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    x = np.arange(n_classes)
    ax.bar(x - 0.2, mean_pred, 0.4, label="Mean predicted probability", color="steelblue")
    if y_true is not None:
        y_true = np.array(y_true)
        mean_true = np.mean((y_true >= 0.5).astype(np.float32), axis=0)
        ax.bar(x + 0.2, mean_true, 0.4, label="Ground-truth prevalence", color="coral", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.set_title("Feature Importance: Mean Prediction & Prevalence by Class")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "feature_importance_by_class.png"), dpi=150, bbox_inches="tight")
    plt.close()


def plot_aggregate_confusion(y_true, y_pred, save_dir, labels=DISEASE_LABELS):
    """Single aggregate multi-label confusion: rows = true label (argmax), cols = pred label (argmax)."""
    os.makedirs(save_dir, exist_ok=True)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    n_classes = y_true.shape[1]
    if len(labels) != n_classes:
        labels = [str(i) for i in range(n_classes)]
    true_idx = np.argmax(y_true, axis=1)
    pred_idx = np.argmax(y_pred, axis=1)
    cm = confusion_matrix(true_idx, pred_idx, labels=range(n_classes))
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", color="black")
    plt.colorbar(im, ax=ax)
    ax.set_title("Aggregate Confusion Matrix (argmax true vs argmax pred)")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "confusion_matrix_aggregate.png"), dpi=150, bbox_inches="tight")
    plt.close()


def run_full_evaluation(
    model,
    X_test,
    y_test,
    history_warmup=None,
    history_finetune=None,
    output_dir="outputs",
    X_val=None,
    y_val=None,
    use_tta=True,
):
    """
    Compute all metrics, save JSON report, and generate all visualizations.
    If X_val/y_val are provided, thresholds are tuned on validation (macro F1 per class), then applied on test.
    use_tta: average predictions over horizontal/vertical flips on test (and val when tuning).
    """
    os.makedirs(output_dir, exist_ok=True)
    eval_dir = os.path.join(output_dir, "evaluation")
    os.makedirs(eval_dir, exist_ok=True)

    y_test = np.array(y_test)
    y_test_bin = (y_test >= 0.5).astype(np.int32) if y_test.dtype in (np.float32, np.float64) else y_test

    print("Computing predictions on test set...")
    _pred = predict_with_tta if use_tta else model.predict
    y_pred_proba = _pred(model, X_test, batch_size=16, verbose=1) if use_tta else model.predict(X_test, verbose=1)

    metrics_050 = compute_all_metrics(y_test, y_pred_proba, threshold=0.5)
    metrics_bundle = {
        "test_metrics_threshold_0.5": metrics_050,
        "use_tta": bool(use_tta),
    }

    thresholds_tuned = None
    if X_val is not None and y_val is not None:
        print("Tuning per-class thresholds on validation set (max F1 per class)...")
        y_val = np.array(y_val)
        y_val_proba = (
            predict_with_tta(model, X_val, batch_size=16, verbose=0)
            if use_tta
            else model.predict(X_val, verbose=0)
        )
        thresholds_tuned = tune_per_class_thresholds_f1(y_val, y_val_proba)
        metrics_bundle["per_class_thresholds_tuned_on_val"] = thresholds_tuned.tolist()
        metrics_tuned = compute_all_metrics(y_test, y_pred_proba, threshold=thresholds_tuned)
        metrics_bundle["test_metrics_threshold_tuned"] = metrics_tuned
        metrics = metrics_tuned
        y_pred = binarize_predictions(y_pred_proba, thresholds_tuned)
        print(f"Tuned thresholds (per class): {np.round(thresholds_tuned, 3)}")
    else:
        metrics = metrics_050
        y_pred = binarize_predictions(y_pred_proba, 0.5)

    with open(os.path.join(eval_dir, "metrics.json"), "w") as f:
        json.dump(
            _json_safe({**metrics_bundle, "primary_reported_test_metrics": metrics}),
            f,
            indent=2,
        )

    print("\n--- Evaluation Metrics (primary) ---")
    print(f"Accuracy:          {metrics['accuracy']:.4f}")
    print(f"Subset Accuracy:  {metrics['subset_accuracy']:.4f}")
    print(f"Precision (macro): {metrics['precision_macro']:.4f}")
    print(f"Recall (macro):   {metrics['recall_macro']:.4f}")
    print(f"F1 (macro):       {metrics['f1_macro']:.4f}")
    print(f"Hamming Loss:     {metrics['hamming_loss']:.4f}")
    print(f"ROC AUC (macro):  {metrics['roc_auc_macro']:.4f}")
    print(f"Avg Precision:    {metrics['average_precision_macro']:.4f}")
    if thresholds_tuned is not None:
        print(f"(Also saved metrics at 0.5 threshold and full bundle in metrics.json)")

    # Training curves (merge both phases if provided)
    if history_warmup is not None:
        plot_training_history(history_warmup, eval_dir, prefix="warmup")
    if history_finetune is not None:
        plot_training_history(history_finetune, eval_dir, prefix="finetune")

    plot_confusion_matrices(y_test_bin, y_pred, eval_dir)
    plot_aggregate_confusion(y_test_bin, y_pred, eval_dir)
    plot_roc_curves(y_test_bin, y_pred_proba, eval_dir)
    plot_pr_curves(y_test_bin, y_pred_proba, eval_dir)
    plot_metrics_summary(metrics, eval_dir)
    plot_feature_importance(y_pred_proba, eval_dir, y_true=y_test_bin)

    print(f"\nAll evaluation plots saved to: {eval_dir}")
    return metrics
