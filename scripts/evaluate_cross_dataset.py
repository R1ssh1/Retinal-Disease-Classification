"""
Cross-dataset evaluation (SDP Future Work).
Evaluates a trained model on a (possibly different) dataset in ODIR-like format.
Usage:
  python scripts/evaluate_cross_dataset.py
  python scripts/evaluate_cross_dataset.py --csv path/to/csv --img_dir path/to/images
"""
import os
import sys
import argparse
import json

# Project root
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import tensorflow as tf

from src.dataset import load_odir_dataset
from src.model import build_concept_aware_lnn
from src.evaluation import run_full_evaluation


def load_saved_model(model_path, input_shape=(224, 224, 3), num_classes=8):
    """Load Keras model; if architecture differs, rebuild and load weights."""
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        return model, None
    except Exception:
        model, last_conv_name, _ = build_concept_aware_lnn(input_shape=input_shape, num_classes=num_classes)
        model.load_weights(model_path)
        return model, last_conv_name


def main():
    parser = argparse.ArgumentParser(description="Cross-dataset evaluation")
    parser.add_argument("--model", type=str, default=os.path.join(ROOT, "models", "concept_lnn_optimal.weights.h5"))
    parser.add_argument("--csv", type=str, default=None, help="Path to CSV (ODIR format with filename, target)")
    parser.add_argument("--img_dir", type=str, default=None, help="Path to image directory")
    parser.add_argument("--output_dir", type=str, default=os.path.join(ROOT, "outputs", "cross_dataset"))
    parser.add_argument("--sample_fraction", type=float, default=1.0)
    args = parser.parse_args()

    # Default: use same ODIR-2 dataset paths as main.py
    if args.csv is None or args.img_dir is None:
        args.csv = os.path.join(ROOT, "dataset", "ODIR-2 dataset", "final_clean_dataset.csv")
        args.img_dir = os.path.join(ROOT, "dataset", "ODIR-2 dataset", "ben_graham_images-20260216T043422Z-3-001", "ben_graham_images")
    if not os.path.isfile(args.csv) or not os.path.isdir(args.img_dir):
        print("CSV or image dir not found. Provide --csv and --img_dir for a different dataset.")
        return 1

    print("Loading model...")
    model, _ = load_saved_model(args.model)
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc", multi_label=True)],
    )

    print("Loading cross-dataset...")
    X_train, X_test, y_train, y_test = load_odir_dataset(args.csv, args.img_dir, sample_fraction=args.sample_fraction)
    print(f"Test set size: {len(X_test)}")

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump({"csv": args.csv, "img_dir": args.img_dir, "model": args.model}, f, indent=2)

    print("Running evaluation...")
    run_full_evaluation(model, X_test, y_test, output_dir=args.output_dir)
    print(f"Results saved to {args.output_dir}")
    return 0


if __name__ == "__main__":
    exit(main())
