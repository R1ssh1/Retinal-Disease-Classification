"""
Data pipeline with configurable augmentation for training (SDP / Future Work).
Use this for training to reduce overfitting; inference can use plain numpy arrays.
"""
import tensorflow as tf
import numpy as np


def augment_image(img, label, training=True):
    """Apply augmentation only when training=True."""
    if not training:
        return img, label
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.rot90(img, k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
    # Brightness/contrast (fundus-safe ranges)
    img = tf.image.random_brightness(img, max_delta=0.18)
    img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
    img = img + tf.random.normal(tf.shape(img), stddev=0.015)
    img = tf.clip_by_value(img, 0.0, 1.0)
    return img, label


def build_tf_dataset(X, y, batch_size=32, shuffle=True, augment=True, repeat=True):
    """
    Build tf.data.Dataset from numpy arrays with optional augmentation.
    X: (N, H, W, C), y: (N, num_classes), both float32.
    """
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(buffer_size=min(1024, len(X)), seed=42)
    if repeat:
        ds = ds.repeat()
    ds = ds.map(
        lambda x, l: augment_image(x, l, training=augment),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def build_tf_dataset_val(X, y, batch_size=32):
    """Validation/test dataset (no shuffle, no repeat, no augmentation)."""
    return tf.data.Dataset.from_tensor_slices((X, y)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
