"""
Interactive demo: upload a fundus image → multi-label predictions + Grad-CAM + XAI text.
Run: flask --app app run  (or python app.py)
Requires trained weights at models/concept_lnn_optimal.weights.h5 (see main.py).

API: POST /api/predict — multipart field "image" or JSON {"image_base64": "..."}.
"""
import base64
import os
import re
import tempfile

import matplotlib

matplotlib.use("Agg")

import cv2
import numpy as np
from flask import Flask, flash, jsonify, redirect, render_template, request, url_for

from src.explainability import DISEASE_LABELS, display_gradcam, make_gradcam_heatmap
from src.llm_explanation import LABELS, build_xai_text_report, generate_explanation
from src.model import build_concept_aware_lnn

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-change-in-production")

_MODEL = None
_LAST_CONV = None
_WEIGHTS_PATH = os.environ.get(
    "MODEL_WEIGHTS",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "concept_lnn_optimal.weights.h5"),
)


def _load_model():
    global _MODEL, _LAST_CONV
    if _MODEL is not None:
        return _MODEL, _LAST_CONV
    model, last_conv_layer_name, _ = build_concept_aware_lnn(input_shape=(224, 224, 3), num_classes=8)
    if not os.path.isfile(_WEIGHTS_PATH):
        raise FileNotFoundError(
            f"No weights at {_WEIGHTS_PATH}. Train with main.py first (saves concept_lnn_optimal.weights.h5)."
        )
    model.load_weights(_WEIGHTS_PATH)
    _MODEL = model
    _LAST_CONV = last_conv_layer_name
    return _MODEL, _LAST_CONV


def _decode_bytes(raw: bytes) -> np.ndarray:
    """RGB float32 (224,224,3) in [0,1], same as src/dataset.load_odir_dataset."""
    if not raw:
        raise ValueError("Empty file.")
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image (use PNG or JPEG fundus photo).")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    return (img.astype(np.float32) / 255.0)


def _decode_upload(file_storage) -> np.ndarray:
    return _decode_bytes(file_storage.read())


def _parse_data_url_b64(s: str) -> bytes:
    s = s.strip()
    m = re.match(r"^data:image/[\w+.-]+;base64,(.+)$", s, re.I)
    if m:
        s = m.group(1)
    return base64.standard_b64decode(s)


def _gradcam_jpeg_b64(img_rgb: np.ndarray, heatmap: np.ndarray, pred_idx: int) -> str:
    fd, path = tempfile.mkstemp(suffix=".jpg")
    try:
        os.close(fd)
        display_gradcam(img_rgb, heatmap, cam_path=path, pred_class=pred_idx)
        with open(path, "rb") as f:
            return base64.standard_b64encode(f.read()).decode("ascii")
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def _run_inference(img: np.ndarray, use_api: bool) -> dict:
    model, last_conv = _load_model()
    batch = np.expand_dims(img, axis=0)
    preds = model.predict(batch, verbose=0)
    probs = preds[0]
    pred_idx = int(np.argmax(probs))

    cam_b64 = None
    gradcam_error = None
    try:
        heatmap = make_gradcam_heatmap(batch, model, last_conv, pred_idx)
        cam_b64 = _gradcam_jpeg_b64(img, heatmap, pred_idx)
    except Exception as e:
        gradcam_error = str(e)

    xai_report = build_xai_text_report(
        probs,
        threshold_positive=0.5,
        gradcam_class_idx=pred_idx,
        shap_class_idx=None,
        sample_index=None,
    )
    extra_nl = generate_explanation(
        probs,
        use_api=use_api,
        gradcam_summary=f"explained class index {pred_idx} ({LABELS[pred_idx]})",
    )

    probabilities = {LABELS[i]: float(probs[i]) for i in range(min(len(LABELS), len(probs)))}
    rows = [{"code": LABELS[i], "prob": float(probs[i])} for i in range(min(len(LABELS), len(probs)))]

    return {
        "probabilities": probabilities,
        "probabilities_list": rows,
        "labels_order": list(LABELS),
        "argmax_index": pred_idx,
        "argmax_code": DISEASE_LABELS[pred_idx] if pred_idx < len(DISEASE_LABELS) else str(pred_idx),
        "xai_report": xai_report,
        "natural_language": extra_nl,
        "gradcam_jpeg_base64": cam_b64,
        "gradcam_error": gradcam_error,
    }


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files or not request.files["image"].filename:
        flash("Please choose an image file.")
        return redirect(url_for("index"))
    use_api = request.form.get("use_llm_api") == "1"
    try:
        img = _decode_upload(request.files["image"])
    except ValueError as e:
        flash(str(e))
        return redirect(url_for("index"))
    try:
        out = _run_inference(img, use_api)
    except FileNotFoundError as e:
        flash(str(e))
        return redirect(url_for("index"))

    if out["gradcam_error"]:
        flash(f"Grad-CAM failed (predictions still shown): {out['gradcam_error']}")

    return render_template(
        "result.html",
        rows=out["probabilities_list"],
        argmax_code=out["argmax_code"],
        xai_report=out["xai_report"],
        extra_explanation=out["natural_language"],
        gradcam_b64=out["gradcam_jpeg_base64"],
        use_api=use_api,
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    Multipart: field "image" (file), optional "use_llm_api" = "1" / "true".
    JSON: {"image_base64": "<raw or data URL>", "use_llm_api": false}
    """
    use_api = False
    raw_bytes = None

    ct = (request.content_type or "").split(";")[0].strip().lower()
    if ct == "application/json":
        data = request.get_json(silent=True) or {}
        use_api = bool(data.get("use_llm_api"))
        b64 = data.get("image_base64")
        if not b64 or not isinstance(b64, str):
            return jsonify(ok=False, error='JSON body must include string "image_base64".'), 400
        try:
            raw_bytes = _parse_data_url_b64(b64)
        except Exception as e:
            return jsonify(ok=False, error=f"Invalid base64 image: {e}"), 400
    else:
        use_api = request.form.get("use_llm_api", "").lower() in ("1", "true", "yes")
        if "image" not in request.files or not request.files["image"].filename:
            return jsonify(ok=False, error='Missing multipart file field "image".'), 400
        raw_bytes = request.files["image"].read()

    try:
        img = _decode_bytes(raw_bytes)
    except ValueError as e:
        return jsonify(ok=False, error=str(e)), 400

    try:
        out = _run_inference(img, use_api)
    except FileNotFoundError as e:
        return jsonify(ok=False, error=str(e)), 503

    payload = {"ok": True, **out}
    return jsonify(payload)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "5000")), debug=True)
