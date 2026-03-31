"""
Explainability AI: Grad-CAM and SHAP for retinal disease classification.
"""
import os
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

from src.llm_explanation import build_xai_text_report

# Disease labels for plots (ODIR-5K)
DISEASE_LABELS = ["N", "D", "G", "C", "A", "H", "M", "O"]


# ---------------------------------------------------------------------------
# Grad-CAM
# ---------------------------------------------------------------------------
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    Generates a Grad-CAM heatmap for a given image and model.
    For multi-label: pass pred_index (int) for the class to explain, or None to use argmax.
    """
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array, training=False)
        if pred_index is None:
            pred_index = int(tf.argmax(preds[0]).numpy())
        # Scalar for gradient (multi-label: explain this class)
        class_score = tf.reduce_sum(preds[:, pred_index])

    grads = tape.gradient(class_score, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_output = last_conv_layer_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def display_gradcam(img, heatmap, cam_path="gradcam.jpg", alpha=0.4, pred_class=None):
    """
    Saves a 3-panel figure: original image, Grad-CAM heatmap, overlay.
    """
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    img_uint8 = np.uint8(255 * np.clip(img, 0, 1))
    if len(img_uint8.shape) == 2:
        img_uint8 = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2BGR)
    elif img_uint8.shape[-1] == 3:
        img_uint8 = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
    superimposed = cv2.addWeighted(img_uint8, 1 - alpha, heatmap_colored, alpha, 0)
    superimposed_rgb = cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title("Original Fundus Image")
    plt.imshow(img)
    plt.axis("off")
    plt.subplot(1, 3, 2)
    plt.title("Grad-CAM Attention Map")
    plt.imshow(cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.subplot(1, 3, 3)
    title = "Grad-CAM Overlay"
    if pred_class is not None and 0 <= pred_class < len(DISEASE_LABELS):
        title += f" (class: {DISEASE_LABELS[pred_class]})"
    plt.title(title)
    plt.imshow(superimposed_rgb)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(cam_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Grad-CAM saved: {cam_path}")


# ---------------------------------------------------------------------------
# SHAP (GradientExplainer)
# ---------------------------------------------------------------------------
def _get_shap():
    try:
        import shap
        return shap
    except ImportError:
        return None


def make_shap_explainer(model, background_data, nsamples=50):
    """
    Builds a SHAP GradientExplainer for the model using background_data.
    background_data: numpy array (n_background, H, W, C), e.g. 20–100 images.
    """
    shap = _get_shap()
    if shap is None:
        raise ImportError("Install SHAP: pip install shap")
    return shap.GradientExplainer(model, background_data, batch_size=min(20, len(background_data)))


def explain_with_shap(explainer, img_batch, class_index=None, nsamples=25):
    """
    Computes SHAP values for img_batch. For multi-label, class_index selects
    which output to explain (default: None = argmax of predictions).
    Returns: shap_values (H, W, C) or (H, W) for one sample, for the chosen class.
    """
    shap = _get_shap()
    if shap is None:
        raise ImportError("Install SHAP: pip install shap")
    shap_values = explainer.shap_values(img_batch, nsamples=nsamples)
    # One input, multiple outputs: shape (batch, H, W, C, num_outputs)
    if isinstance(shap_values, tuple):
        shap_values = shap_values[0]
    sv = np.array(shap_values)
    if class_index is None:
        preds = explainer.model.predict(img_batch, verbose=0)
        class_index = int(np.argmax(preds[0]))
    if sv.ndim == 5:
        sv = sv[..., class_index]
    if sv.ndim == 4 and sv.shape[0] == 1:
        sv = sv[0]
    return sv


def display_shap(img, shap_values, save_path="shap.jpg", class_index=None):
    """
    Saves SHAP visualization: original image and SHAP heatmap (mean over channels).
    shap_values: (H, W, C) or (H, W); img: (H, W, C) in [0,1].
    """
    if shap_values.ndim == 3:
        shap_2d = np.mean(np.abs(shap_values), axis=-1)
    else:
        shap_2d = np.squeeze(shap_values)
    shap_2d = (shap_2d - shap_2d.min()) / (shap_2d.max() - shap_2d.min() + 1e-8)
    shap_resized = cv2.resize(shap_2d, (img.shape[1], img.shape[0]))
    shap_colored = cv2.applyColorMap(np.uint8(255 * shap_resized), cv2.COLORMAP_JET)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.title("Input Image")
    plt.imshow(img)
    plt.axis("off")
    plt.subplot(1, 2, 2)
    title = "SHAP Importance"
    if class_index is not None and 0 <= class_index < len(DISEASE_LABELS):
        title += f" (class {DISEASE_LABELS[class_index]})"
    plt.title(title)
    plt.imshow(cv2.cvtColor(shap_colored, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"SHAP saved: {save_path}")


def plot_shap_bar(preds, shap_values_per_class, save_path="shap_bar.jpg"):
    """
    Bar plot of mean |SHAP| per output class (feature importance for each disease).
    preds: (n_classes,); shap_values_per_class: list of arrays or single (n_classes,) mean abs SHAP.
    """
    shap = _get_shap()
    if shap is None:
        return
    if not isinstance(shap_values_per_class, (list, tuple)):
        mean_abs = shap_values_per_class
    else:
        mean_abs = [np.mean(np.abs(s)) for s in shap_values_per_class]
    mean_abs = np.array(mean_abs).flatten()
    n = min(len(DISEASE_LABELS), len(mean_abs))
    labels = DISEASE_LABELS[:n]
    plt.figure(figsize=(10, 4))
    plt.bar(labels, mean_abs[:n], color="steelblue", edgecolor="navy")
    plt.xlabel("Disease / Class")
    plt.ylabel("Mean |SHAP|")
    plt.title("SHAP Feature Importance by Class")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"SHAP bar plot saved: {save_path}")


def run_full_explainability(model, last_conv_layer_name, X_test, output_dir="outputs", n_gradcam=3, n_shap_background=30, n_shap_explain=3):
    """
    Runs both Grad-CAM and SHAP for a few test samples and saves all visualizations
    under output_dir (gradcam and shap subfolders).
    """
    os.makedirs(output_dir, exist_ok=True)
    gradcam_dir = os.path.join(output_dir, "gradcam")
    shap_dir = os.path.join(output_dir, "shap")
    os.makedirs(gradcam_dir, exist_ok=True)
    os.makedirs(shap_dir, exist_ok=True)
    xai_dir = os.path.join(output_dir, "xai")
    os.makedirs(xai_dir, exist_ok=True)

    # Grad-CAM + text report (explains model output + what each map refers to)
    for i in range(min(n_gradcam, len(X_test))):
        img = X_test[i]
        batch = np.expand_dims(img, axis=0)
        preds = model.predict(batch, verbose=0)
        pred_idx = int(np.argmax(preds[0]))
        try:
            heatmap = make_gradcam_heatmap(batch, model, last_conv_layer_name, pred_idx)
            display_gradcam(
                img, heatmap,
                cam_path=os.path.join(gradcam_dir, f"gradcam_sample_{i}_class_{DISEASE_LABELS[pred_idx]}.jpg"),
                pred_class=pred_idx
            )
            report = build_xai_text_report(
                preds[0],
                threshold_positive=0.5,
                gradcam_class_idx=pred_idx,
                shap_class_idx=pred_idx,
                sample_index=i,
            )
            report_path = os.path.join(xai_dir, f"explanation_sample_{i}.txt")
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(report)
            print(f"XAI text report saved: {report_path}")
        except Exception as e:
            print(f"Grad-CAM sample {i} failed: {e}")

    # SHAP (optional; can be slow)
    shap = _get_shap()
    if shap is not None and len(X_test) >= n_shap_background:
        bg = X_test[:n_shap_background]
        try:
            explainer = make_shap_explainer(model, bg, nsamples=20)
            for i in range(min(n_shap_explain, len(X_test))):
                batch = np.expand_dims(X_test[i], axis=0)
                preds = model.predict(batch, verbose=0)
                pred_idx = int(np.argmax(preds[0]))
                sv = explain_with_shap(explainer, batch, class_index=pred_idx, nsamples=15)
                display_shap(
                    X_test[i], np.squeeze(sv),
                    save_path=os.path.join(shap_dir, f"shap_sample_{i}_class_{DISEASE_LABELS[pred_idx]}.jpg"),
                    class_index=pred_idx
                )
        except Exception as e:
            print(f"SHAP explanation failed (install shap if needed): {e}")
    else:
        if shap is None:
            print("SHAP skipped (pip install shap).")
        else:
            print("SHAP skipped (need enough test samples for background).")
