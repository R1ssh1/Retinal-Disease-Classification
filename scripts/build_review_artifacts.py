"""
Build review artifacts from available project outputs.

This script is intentionally lightweight and uses only stdlib so it can run
even when ML dependencies are unavailable in the local environment.
"""
import json
import os
from datetime import datetime


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METRICS_PATH = os.path.join(ROOT, "outputs", "evaluation", "metrics.json")
OUT_DIR = os.path.join(ROOT, "outputs", "review")


def _read_metrics(path):
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    # New format: bundle with primary_reported_test_metrics
    if isinstance(raw, dict) and "primary_reported_test_metrics" in raw:
        return raw["primary_reported_test_metrics"]
    return raw


def _fmt(v):
    if isinstance(v, (int, float)):
        return f"{v:.4f}"
    return str(v)


def write_metrics_markdown(metrics, out_path):
    lines = []
    lines.append("# Review Metrics Snapshot")
    lines.append("")
    lines.append(f"- Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"- Source: `outputs/evaluation/metrics.json`")
    lines.append("")
    if not metrics:
        lines.append("No metrics file found. Run evaluation first.")
    else:
        rows = [
            ("accuracy", "Accuracy"),
            ("subset_accuracy", "Subset Accuracy"),
            ("precision_macro", "Precision (Macro)"),
            ("recall_macro", "Recall (Macro)"),
            ("f1_macro", "F1 (Macro)"),
            ("precision_micro", "Precision (Micro)"),
            ("recall_micro", "Recall (Micro)"),
            ("f1_micro", "F1 (Micro)"),
            ("roc_auc_macro", "ROC-AUC (Macro)"),
            ("roc_auc_micro", "ROC-AUC (Micro)"),
            ("average_precision_macro", "Average Precision (Macro)"),
            ("average_precision_micro", "Average Precision (Micro)"),
            ("hamming_loss", "Hamming Loss"),
        ]
        lines.append("| Metric | Value |")
        lines.append("|---|---:|")
        for key, label in rows:
            lines.append(f"| {label} | {_fmt(metrics.get(key, 'N/A'))} |")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def write_classwise_markdown(metrics, out_path):
    p = metrics.get("precision_per_class", []) if metrics else []
    r = metrics.get("recall_per_class", []) if metrics else []
    f1 = metrics.get("f1_per_class", []) if metrics else []
    labels = ["N", "D", "G", "C", "A", "H", "M", "O"]
    n = min(len(labels), len(p), len(r), len(f1))

    lines = []
    lines.append("# Class-wise Metrics Snapshot")
    lines.append("")
    lines.append("| Class | Precision | Recall | F1 |")
    lines.append("|---|---:|---:|---:|")
    if n == 0:
        lines.append("| N/A | N/A | N/A | N/A |")
    else:
        for i in range(n):
            lines.append(f"| {labels[i]} | {_fmt(p[i])} | {_fmt(r[i])} | {_fmt(f1[i])} |")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def write_summary_json(metrics, out_path):
    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_metrics": "outputs/evaluation/metrics.json",
        "status": "preliminary" if metrics else "missing_metrics",
        "key_points": [],
    }
    if metrics:
        summary["key_points"] = [
            f"Macro F1: {_fmt(metrics.get('f1_macro'))}",
            f"Micro F1: {_fmt(metrics.get('f1_micro'))}",
            f"ROC-AUC macro: {_fmt(metrics.get('roc_auc_macro'))}",
            f"Hamming loss: {_fmt(metrics.get('hamming_loss'))}",
        ]
    else:
        summary["key_points"] = ["Run the evaluation pipeline to generate metrics."]

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    metrics = _read_metrics(METRICS_PATH)

    write_metrics_markdown(metrics, os.path.join(OUT_DIR, "review_metrics_table.md"))
    write_classwise_markdown(metrics, os.path.join(OUT_DIR, "review_classwise_table.md"))
    write_summary_json(metrics, os.path.join(OUT_DIR, "review_summary.json"))

    print(f"Review artifacts generated at: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

