"""
LLM-based medical explanation generation (SDP Future Work).
Produces natural-language explanations for model predictions using:
1) Template-based explanations (no API, offline).
2) Optional OpenAI-compatible API for richer text (set OPENAI_API_KEY if desired).
"""
import os
import re

import numpy as np

# ODIR disease names for readable explanations
DISEASE_NAMES = {
    "N": "No abnormality",
    "D": "Diabetic Retinopathy",
    "G": "Glaucoma",
    "C": "Cataract",
    "A": "Age-related Macular Degeneration",
    "H": "Hypertension",
    "M": "Myopia",
    "O": "Other abnormalities",
}

LABELS = ["N", "D", "G", "C", "A", "H", "M", "O"]


def _template_explanation(pred_probs, top_k=3, threshold=0.3):
    """
    Build a short clinical-style explanation from prediction probabilities.
    pred_probs: (8,) or list of 8 floats (sigmoid outputs).
    """
    pred_probs = list(pred_probs)[:8]
    indexed = [(LABELS[i], pred_probs[i], DISEASE_NAMES.get(LABELS[i], LABELS[i])) for i in range(len(LABELS))]
    indexed.sort(key=lambda x: -x[1])
    positive = [(lbl, p, name) for lbl, p, name in indexed if p >= threshold][:top_k]
    if not positive:
        positive = [indexed[0]]
    lines = [
        "The model indicates the following findings (probabilities in parentheses):"
    ]
    for lbl, p, name in positive:
        pct = round(100 * p, 1)
        lines.append(f"  • {name} ({pct}%)")
    lines.append("\nThis is a screening aid only. Please consult an ophthalmologist for diagnosis.")
    return "\n".join(lines)


def generate_explanation(pred_probs, use_api=False, image_path=None, gradcam_summary=None):
    """
    Generate natural-language explanation for a single prediction.
    pred_probs: array/list of length 8 (sigmoid outputs).
    use_api: if True and OPENAI_API_KEY is set, call OpenAI (or compatible) API for a longer explanation.
    image_path, gradcam_summary: optional context for API call.
    Returns: string explanation.
    """
    explanation = _template_explanation(pred_probs)
    if not use_api:
        return explanation
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return explanation
    try:
        import openai
        if not hasattr(openai, "OpenAI"):
            return explanation
        client = openai.OpenAI(api_key=api_key)
        prompt = (
            "You are a medical AI assistant. Based on the following retinal screening model output, "
            "write 2-3 short, clear sentences for a clinician. Do not diagnose; state only what the model suggests.\n\n"
            f"Model output (probabilities per condition): {dict(zip(LABELS, [round(float(p), 3) for p in pred_probs]))}\n\n"
            + (f"Grad-CAM summary: {gradcam_summary}\n" if gradcam_summary else "")
            + "Reply in plain text only."
        )
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
        )
        if getattr(resp, "choices", None) and len(resp.choices) > 0:
            text = getattr(resp.choices[0].message, "content", "") or ""
            if text.strip():
                return text.strip() + "\n\n[Template fallback]\n" + explanation
    except Exception:
        pass
    return explanation


def generate_explanation_for_image(model, image_batch, class_names=LABELS):
    """
    Convenience: run model on one image batch, return template explanation for the first sample.
    image_batch: (1, H, W, 3) numpy array, normalized [0,1].
    """
    preds = model.predict(image_batch, verbose=0)
    probs = preds[0]
    return generate_explanation(probs, use_api=False), probs


def build_xai_text_report(
    pred_probs,
    threshold_positive=0.5,
    gradcam_class_idx=None,
    shap_class_idx=None,
    sample_index=None,
):
    """
    Human-readable XAI report that ties **model output** to **what each visualization explains**.

    pred_probs: length-8 sigmoid probabilities (N, D, G, C, A, H, M, O).
    gradcam_class_idx / shap_class_idx: which output neuron the heatmaps refer to (usually argmax or clinician-chosen).
    """
    pred_probs = list(np.asarray(pred_probs, dtype=np.float64).flatten()[:8])
    lines = []
    title = "=== Retinal screening model — explanation of output ==="
    if sample_index is not None:
        title += f" (sample {sample_index})"
    lines.append(title)
    lines.append("")
    lines.append("1) MODEL OUTPUT (probability per condition, 0–100%)")
    lines.append("   Codes: N=Normal, D=Diabetic Retinopathy, G=Glaucoma, C=Cataract,")
    lines.append("          A=AMD, H=Hypertension, M=Myopia, O=Other")
    lines.append("")
    for i, lbl in enumerate(LABELS):
        name = DISEASE_NAMES.get(lbl, lbl)
        pct = round(100.0 * float(pred_probs[i]), 1)
        flag = "  [above threshold]" if pred_probs[i] >= threshold_positive else ""
        lines.append(f"   • {lbl} ({name}): {pct}%{flag}")
    lines.append("")
    pos = [
        (LABELS[i], pred_probs[i], DISEASE_NAMES.get(LABELS[i], LABELS[i]))
        for i in range(len(LABELS))
        if pred_probs[i] >= threshold_positive
    ]
    if pos:
        lines.append(
            f"2) INTERPRETATION: conditions at or above {threshold_positive:.0%} probability (multi-label possible)"
        )
        for lbl, p, name in sorted(pos, key=lambda x: -x[1]):
            lines.append(f"   • {name} ({lbl}): {round(100 * p, 1)}%")
    else:
        lines.append(
            f"2) INTERPRETATION: no condition reaches {threshold_positive:.0%}; report highest scores only."
        )
        top = sorted(
            [(LABELS[i], pred_probs[i], DISEASE_NAMES.get(LABELS[i], LABELS[i])) for i in range(len(LABELS))],
            key=lambda x: -x[1],
        )[:3]
        for lbl, p, name in top:
            lines.append(f"   • {name} ({lbl}): {round(100 * p, 1)}%")
    lines.append("")
    lines.append("3) GRAD-CAM (spatial explanation)")
    if gradcam_class_idx is not None and 0 <= gradcam_class_idx < len(LABELS):
        gl = LABELS[gradcam_class_idx]
        gname = DISEASE_NAMES.get(gl, gl)
        lines.append(
            f"   The heatmap shows **where** the network focused for output neuron "
            f"'{gl}' ({gname}) — i.e., regions that most influenced that score."
        )
        lines.append(
            "   Compare the overlay to optic disc, vessels, and macula in the fundus image."
        )
    else:
        lines.append("   (No class index provided for Grad-CAM.)")
    lines.append("")
    lines.append("4) SHAP (attribution explanation)")
    if shap_class_idx is not None and 0 <= shap_class_idx < len(LABELS):
        sl = LABELS[shap_class_idx]
        sname = DISEASE_NAMES.get(sl, sl)
        lines.append(
            f"   The map summarizes **pixel-level contributions** toward neuron '{sl}' ({sname})."
        )
        lines.append("   Higher intensity ≈ stronger push toward that class in this sample.")
    else:
        lines.append("   (No class index provided for SHAP.)")
    lines.append("")
    lines.append("5) SUMMARY (template)")
    lines.append(_template_explanation(pred_probs, top_k=4, threshold=0.25))
    lines.append("")
    lines.append(
        "Disclaimer: This is an AI screening aid, not a diagnosis. "
        "Clinical correlation by an ophthalmologist is required."
    )
    return "\n".join(lines)
