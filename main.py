import os
import argparse
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from src.dataset import load_odir_dataset
from src.model import build_concept_aware_lnn
from src.explainability import run_full_explainability
from src.evaluation import run_full_evaluation
from src.data_pipeline import build_tf_dataset, build_tf_dataset_val
from sklearn.model_selection import train_test_split
from src.losses import (
    make_weighted_binary_crossentropy,
    make_weighted_focal_loss,
    compute_pos_weights,
)
import numpy as np


def _split_train_val(X_train, y_train, val_fraction=0.15):
    """Hold out a validation set from training data for early stopping and threshold tuning."""
    strat = np.argmax(y_train, axis=1)
    try:
        return train_test_split(
            X_train, y_train, test_size=val_fraction, random_state=42, stratify=strat
        )
    except ValueError:
        return train_test_split(X_train, y_train, test_size=val_fraction, random_state=42)

# Dataset paths: prefer local dataset/ (supports "ODIR-2 dataset" folder structure)
_BASE = os.path.dirname(os.path.abspath(__file__))
_ODIR2_CSV = os.path.join(_BASE, "dataset", "ODIR-2 dataset", "final_clean_dataset.csv")
_ODIR2_IMG = os.path.join(_BASE, "dataset", "ODIR-2 dataset", "ben_graham_images-20260216T043422Z-3-001", "ben_graham_images")
_LOCAL_CSV = os.path.join(_BASE, "dataset", "final_clean_dataset.csv")
_LOCAL_IMG = os.path.join(_BASE, "dataset", "ben_graham_images")
KAGDLE_CSV = "/kaggle/input/datasets/swatinandha/odir-5k/final_clean_dataset.csv"
KAGDLE_IMG = "/kaggle/input/datasets/swatinandha/odir-5k/ben_graham_images-20260216T043422Z-3-001/ben_graham_images/"

def _get_data_paths():
    if os.path.isfile(_ODIR2_CSV) and os.path.isdir(_ODIR2_IMG):
        return _ODIR2_CSV, _ODIR2_IMG
    if os.path.isfile(_LOCAL_CSV) and os.path.isdir(_LOCAL_IMG):
        return _LOCAL_CSV, _LOCAL_IMG
    if os.path.isfile(KAGDLE_CSV) and os.path.isdir(KAGDLE_IMG):
        return KAGDLE_CSV, KAGDLE_IMG
    return None, None

def setup_mock_data():
    """Generates synthetic dataset arrays for initial checks/development."""
    print("Warning: Real data paths not found locally. Running a quick dry-run with mock data.")
    mock_X_train = np.random.rand(20, 224, 224, 3).astype(np.float32)
    mock_X_test = np.random.rand(10, 224, 224, 3).astype(np.float32)
    mock_y_train = np.random.randint(0, 2, size=(20, 8)).astype(np.float32)
    mock_y_test = np.random.randint(0, 2, size=(10, 8)).astype(np.float32)
    return mock_X_train, mock_X_test, mock_y_train, mock_y_test


def _merge_histories(history_warmup, history_finetune):
    """
    Merge two Keras History objects into one lightweight history-like object
    so evaluation can produce single continuous plots.
    """
    merged = type("MergedHistory", (), {})()
    merged.history = {}
    keys = set(history_warmup.history.keys()) | set(history_finetune.history.keys())
    for key in keys:
        merged.history[key] = history_warmup.history.get(key, []) + history_finetune.history.get(key, [])
    return merged


def parse_args():
    parser = argparse.ArgumentParser(description="Train/evaluate retinal disease model")
    parser.add_argument("--sample_fraction", type=float, default=1.0, help="Fraction of dataset to load (0,1].")
    parser.add_argument("--warmup_epochs", type=int, default=15, help="Epochs for warm-up phase.")
    parser.add_argument("--finetune_epochs", type=int, default=30, help="Epochs for fine-tuning phase.")
    parser.add_argument("--warmup_batch_size", type=int, default=32, help="Batch size for warm-up.")
    parser.add_argument("--finetune_batch_size", type=int, default=16, help="Batch size for fine-tuning.")
    parser.add_argument("--mock_data", action="store_true", help="Force synthetic mock data for smoke testing.")
    parser.add_argument("--skip_explainability", action="store_true", help="Skip Grad-CAM/SHAP generation.")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory for evaluation and plots.")
    parser.add_argument(
        "--no_class_weights",
        action="store_true",
        help="Use plain BCE instead of class-balanced BCE (for ablation).",
    )
    parser.add_argument(
        "--unfreeze_layers",
        type=int,
        default=4,
        help="Number of ResNet50V2 top layers to unfreeze in fine-tuning (smaller = less overfitting).",
    )
    parser.add_argument(
        "--lnn_units",
        type=str,
        default="256,128",
        help="Comma-separated Liquid layer units, e.g., '256,128'. Smaller head often generalizes better.",
    )
    parser.add_argument(
        "--disable_model_augmentation",
        action="store_true",
        help="Disable in-model augmentation and rely only on tf.data augmentation (recommended for stability).",
    )
    parser.add_argument(
        "--early_stop_metric",
        type=str,
        choices=["val_auc", "val_loss"],
        default="val_auc",
        help="Metric for EarlyStopping/ModelCheckpoint. val_auc often tracks generalization better on imbalanced labels.",
    )
    parser.add_argument(
        "--label_smoothing",
        type=float,
        default=0.08,
        help="Label smoothing for BCE/focal loss.",
    )
    parser.add_argument(
        "--loss",
        type=str,
        choices=["focal", "bce"],
        default="focal",
        help="focal = weighted focal loss (default, often better on imbalanced multi-label); bce = weighted BCE.",
    )
    parser.add_argument(
        "--val_fraction",
        type=float,
        default=0.15,
        help="Fraction of training split held out for validation + threshold tuning.",
    )
    parser.add_argument("--no_tta", action="store_true", help="Disable test-time augmentation at evaluation.")
    return parser.parse_args()


def _parse_lnn_units(units_text):
    try:
        units = tuple(int(x.strip()) for x in units_text.split(",") if x.strip())
        units = tuple(u for u in units if u > 0)
        return units if units else (256, 128)
    except Exception:
        return (256, 128)


def main():
    args = parse_args()
    os.makedirs("models", exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    print("=======================================================================")
    print("=== Optimized Concept-Aware Retinal Disease Classification (LNN) ===")
    print("=======================================================================\n")

    # 1. Provide Real ODIR Training Pipeline Setup
    print("[1] Loading Data...")
    csv_path, img_dir = _get_data_paths()
    try:
        if (not args.mock_data) and csv_path and img_dir:
            X_train, X_test, y_train, y_test = load_odir_dataset(
                csv_path, img_dir, sample_fraction=args.sample_fraction
            )
        else:
            raise FileNotFoundError("Mock mode enabled or dataset path missing.")
    except Exception as e:
        print(f"Directory Loading Failed. Using Mock Data for system verification. (Error: {e})")
        X_train, X_test, y_train, y_test = setup_mock_data()
        
    print(f"Dataset Split: Train={X_train.shape[0]} images, Test={X_test.shape[0]} images.")

    X_tr, X_val, y_tr, y_val = _split_train_val(X_train, y_train, val_fraction=args.val_fraction)
    print(
        f"Train/val (from train split): {X_tr.shape[0]} train, {X_val.shape[0]} val "
        f"(val used for early stopping + per-class threshold tuning).\n"
    )

    print("[2] Assembling Optimized Model Architecture...")
    lnn_units = _parse_lnn_units(args.lnn_units)
    monitor_metric = args.early_stop_metric
    monitor_mode = "max" if monitor_metric == "val_auc" else "min"

    # 8 Classes per the Retinal dataset targets
    model, last_conv_layer_name, base_cnn = build_concept_aware_lnn(
        input_shape=(224, 224, 3),
        num_classes=8,
        lnn_units=lnn_units,
        use_model_augmentation=not args.disable_model_augmentation,
    )
    print(
        f"Architecture settings: lnn_units={lnn_units}, "
        f"model_augmentation={'off' if args.disable_model_augmentation else 'on'}, "
        f"early_stop_metric={monitor_metric}."
    )

    # Loss: focal (default) or BCE, with class rebalancing on positives
    label_smoothing = max(0.0, min(args.label_smoothing, 0.2))
    if args.no_class_weights:
        train_loss = tf.keras.losses.BinaryCrossentropy(label_smoothing=label_smoothing)
    else:
        pos_w = compute_pos_weights(y_tr)
        print(f"Class pos_weights (neg/pos, clipped): {np.round(pos_w, 2)}")
        if args.loss == "focal":
            train_loss = make_weighted_focal_loss(pos_w, gamma=2.0, label_smoothing=label_smoothing)
            print("Using weighted focal loss (gamma=2).")
        else:
            train_loss = make_weighted_binary_crossentropy(pos_w, label_smoothing=label_smoothing)
            print("Using weighted binary cross-entropy.")

    # Define optimization tracking metrics
    metrics = [
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc', multi_label=True)
    ]
    
    # --- PHASE 1: WARM-UP (Train Only Custom Layers) ---
    print("\n[3] Phase 1: Warm-up Training (Freezing CNN Backbone)...")
    
    # Default AdamW weight decay reduces overfitting (override with WEIGHT_DECAY=0 to disable)
    weight_decay = float(os.environ.get("WEIGHT_DECAY", "1e-4"))
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=1e-3, weight_decay=weight_decay, clipnorm=1.0
    )
    model.compile(
        optimizer=optimizer,
        loss=train_loss,
        metrics=metrics
    )
    
    callbacks_warmup = [
        EarlyStopping(monitor=monitor_metric, patience=4, restore_best_weights=True, mode=monitor_mode),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-5),
        ModelCheckpoint(
            'models/warmup_best.weights.h5',
            save_best_only=True,
            monitor=monitor_metric,
            mode=monitor_mode,
            save_weights_only=True,
        )
    ]

    train_ds_warmup = build_tf_dataset(
        X_tr, y_tr, batch_size=args.warmup_batch_size, shuffle=True, augment=True, repeat=False
    )
    val_ds_warmup = build_tf_dataset_val(X_val, y_val, batch_size=args.warmup_batch_size)

    history_warmup = model.fit(
        train_ds_warmup,
        validation_data=val_ds_warmup,
        epochs=args.warmup_epochs,
        callbacks=callbacks_warmup,
        verbose=1
    )
    
    # --- PHASE 2: FINE-TUNING (Unfreeze Top Layers of CNN) ---
    print("\n[4] Phase 2: Fine-Tuning Training (Unfreezing top CNN layers)...")
    
    # Unfreeze the base CNN
    base_cnn.trainable = True
    
    # Freeze the bottom layers and leave the top few unlocked for fine-tuning (fewer layers = less overfitting)
    n_unfreeze = max(1, min(args.unfreeze_layers, len(base_cnn.layers)))
    for layer in base_cnn.layers[:-n_unfreeze]:
        layer.trainable = False
        
    print(f"Unfrozen the last {n_unfreeze} layers of ResNet50V2 for fine-tuning.")
    
    optimizer_ft = tf.keras.optimizers.AdamW(
        learning_rate=1e-5, weight_decay=weight_decay, clipnorm=1.0
    )
    model.compile(
        optimizer=optimizer_ft,
        loss=train_loss,
        metrics=metrics
    )
    
    callbacks_finetune = [
        EarlyStopping(monitor=monitor_metric, patience=6, restore_best_weights=True, mode=monitor_mode),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7),
        ModelCheckpoint(
            'models/concept_lnn_optimal.weights.h5',
            save_best_only=True,
            monitor=monitor_metric,
            mode=monitor_mode,
            save_weights_only=True,
        )
    ]

    train_ds_finetune = build_tf_dataset(
        X_tr, y_tr, batch_size=args.finetune_batch_size, shuffle=True, augment=True, repeat=False
    )
    val_ds_finetune = build_tf_dataset_val(X_val, y_val, batch_size=args.finetune_batch_size)

    # We resume training
    history_finetune = model.fit(
        train_ds_finetune,
        validation_data=val_ds_finetune,
        epochs=args.finetune_epochs,
        callbacks=callbacks_finetune,
        verbose=1
    )
    
    print("\n[5] Optimal weights saved to 'models/concept_lnn_optimal.weights.h5'")

    print("\n[6] Running Full Evaluation (metrics + visualizations)...")
    run_full_evaluation(
        model,
        X_test,
        y_test,
        history_warmup=_merge_histories(history_warmup, history_finetune),
        history_finetune=None,
        output_dir=args.output_dir,
        X_val=X_val,
        y_val=y_val,
        use_tta=not args.no_tta,
    )

    if not args.skip_explainability:
        print("\n[7] Generating Explainability (Grad-CAM + SHAP)...")
        run_full_explainability(
            model,
            last_conv_layer_name,
            X_test,
            output_dir=args.output_dir,
            n_gradcam=3,
            n_shap_background=30,
            n_shap_explain=3,
        )
    else:
        print("\n[7] Explainability skipped (--skip_explainability enabled).")

    print("\n=======================================================================")
    print("Optimization and Training Completed!")

if __name__ == "__main__":
    main()
