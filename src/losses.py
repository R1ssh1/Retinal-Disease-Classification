"""
Losses for multi-label classification with class imbalance (reduces overfitting to majority patterns).
"""
import tensorflow as tf


def make_weighted_binary_crossentropy(pos_weights, label_smoothing=0.1):
    """
    Binary cross-entropy with per-class positive weights (Keras-style pos_weight).
    pos_weights: shape (num_classes,) — typically neg_count/pos_count per class, clipped.
    """
    pos_w = tf.constant(pos_weights, dtype=tf.float32)

    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        bce = tf.keras.losses.binary_crossentropy(
            y_true, y_pred, from_logits=False, label_smoothing=label_smoothing
        )
        # Weight positive examples more when class is rare: w = y*(pw-1)+1
        w = y_true * (pos_w - 1.0) + 1.0
        return tf.reduce_mean(bce * w)

    return loss


def compute_pos_weights(y_train, max_ratio=20.0):
    """
    Per-class pos_weight = neg/pos, clipped to avoid extreme gradients.
    y_train: (N, C) binary/float in {0,1}.
    """
    import numpy as np

    y = np.asarray(y_train, dtype=np.float64)
    pos = np.maximum(y.sum(axis=0), 1.0)
    n = float(y.shape[0])
    neg = n - y.sum(axis=0)
    pw = neg / pos
    pw = np.clip(pw, 1.0, max_ratio)
    return pw.astype(np.float32)


def make_weighted_focal_loss(pos_weights, gamma=2.0, label_smoothing=0.1):
    """
    Focal loss on sigmoid outputs + per-class positive weights (helps rare classes and hard examples).
    Lin et al. focal: (1 - p_t)^gamma * BCE, with class rebalancing on positives.
    """
    pos_w = tf.constant(pos_weights, dtype=tf.float32)

    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        if label_smoothing > 0:
            y_true = y_true * (1.0 - label_smoothing) + 0.5 * label_smoothing
        p = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        bce = -y_true * tf.math.log(p) - (1.0 - y_true) * tf.math.log(1.0 - p)
        p_t = y_true * p + (1.0 - y_true) * (1.0 - p)
        focal = tf.pow(1.0 - p_t, gamma) * bce
        w = y_true * (pos_w - 1.0) + 1.0
        return tf.reduce_mean(focal * w)

    return loss
