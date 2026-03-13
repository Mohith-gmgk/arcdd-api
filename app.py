from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image
import io
import os
import time

app = Flask(__name__)
CORS(app)

# ─── Lazy model loading ───────────────────────────────────────────
# Load model only on first request to avoid gunicorn startup timeout
MODEL_PATH = os.environ.get("MODEL_PATH", "model.keras")
_model = None

def get_model():
    global _model
    if _model is None:
        print("⏳ Loading model...")
        import tensorflow as tf
        _model = tf.keras.models.load_model(MODEL_PATH)
        print("✅ Model loaded!")
    return _model

# ─── 44 Class Names (exact order from training) ──────────────────
CLASS_NAMES = [
    "apple__apple_scab",
    "apple__black_rot",
    "apple__cedar_apple_rust",
    "apple__healthy",
    "cassava__bacterial_blight_cbb",
    "cassava__brown_streak_disease_cbsd",
    "cassava__green_mottle_cgm",
    "cassava__healthy",
    "cassava__mosaic_disease_cmd",
    "cherry_including_sour__healthy",
    "cherry_including_sour__powdery_mildew",
    "corn_maize__cercospora_leaf_spot_gray_leaf_spot",
    "corn_maize__common_rust",
    "corn_maize__healthy",
    "corn_maize__northern_leaf_blight",
    "grape__black_rot",
    "grape__esca_black_measles",
    "grape__healthy",
    "grape__leaf_blight_isariopsis_leaf_spot",
    "orange__haunglongbing_citrus_greening",
    "peach__bacterial_spot",
    "peach__healthy",
    "pepper_bell__bacterial_spot",
    "pepper_bell__healthy",
    "potato__early_blight",
    "potato__healthy",
    "potato__late_blight",
    "rice__brownspot",
    "rice__healthy",
    "rice__hispa",
    "rice__leafblast",
    "squash__powdery_mildew",
    "strawberry__healthy",
    "strawberry__leaf_scorch",
    "tomato__bacterial_spot",
    "tomato__early_blight",
    "tomato__healthy",
    "tomato__late_blight",
    "tomato__leaf_mold",
    "tomato__septoria_leaf_spot",
    "tomato__spider_mites_two-spotted_spider_mite",
    "tomato__target_spot",
    "tomato__tomato_mosaic_virus",
    "tomato__tomato_yellow_leaf_curl_virus",
]

IMG_SIZE = (224, 224)

def format_class_name(raw):
    parts = raw.split("__")
    crop    = parts[0].replace("_", " ").title()
    disease = parts[1].replace("_", " ").replace("-", " ").title() if len(parts) > 1 else ""
    return crop, disease

@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "ok", "model": "EfficientNetB2", "classes": len(CLASS_NAMES)})

@app.route("/warmup", methods=["GET"])
def warmup():
    try:
        get_model()
        return jsonify({"status": "ready", "model": "EfficientNetB2"})
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    file = request.files["image"]
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400
    try:
        start = time.time()
        model = get_model()
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img = img.resize(IMG_SIZE)
        img_array = np.array(img, dtype=np.float32)
        from tensorflow.keras.applications.efficientnet import preprocess_input
        img_input = preprocess_input(np.expand_dims(img_array, axis=0))
        preds = model.predict(img_input, verbose=0)[0]
        elapsed = round((time.time() - start) * 1000)
        top_idx  = int(np.argmax(preds))
        top_conf = float(preds[top_idx])
        crop, disease = format_class_name(CLASS_NAMES[top_idx])
        top5_indices = np.argsort(preds)[::-1][:5]
        top5 = []
        for i in top5_indices:
            c, d = format_class_name(CLASS_NAMES[i])
            top5.append({"raw": CLASS_NAMES[i], "crop": c, "disease": d, "confidence": float(preds[i])})
        is_healthy = "healthy" in CLASS_NAMES[top_idx].lower()
        if is_healthy:
            severity = "None"
        elif top_conf >= 0.85:
            severity = "High"
        elif top_conf >= 0.55:
            severity = "Medium"
        else:
            severity = "Low"
        print(f"✅ Predicted: {disease} ({crop}) — {top_conf:.1%} in {elapsed}ms")
        return jsonify({
            "success": True,
            "prediction": {
                "raw": CLASS_NAMES[top_idx], "crop": crop, "disease": disease,
                "confidence": top_conf, "severity": severity, "is_healthy": is_healthy,
            },
            "top5": top5,
            "model": "EfficientNetB2",
            "inference_time_ms": elapsed,
        })
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
