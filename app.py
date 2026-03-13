from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image
import io
import os
import time

app = Flask(__name__)
CORS(app)

# ─── TFLite Interpreter ───────────────────────────────────────────
MODEL_PATH = os.environ.get("MODEL_PATH", "model.tflite")
_interpreter = None

def get_interpreter():
    global _interpreter
    if _interpreter is None:
        print("⏳ Loading TFLite model...")
        import tensorflow as tf
        _interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        _interpreter.allocate_tensors()
        print("✅ TFLite model loaded!")
    return _interpreter

# Preload in background thread
import threading
threading.Thread(target=get_interpreter, daemon=True).start()

# ─── 44 Class Names ───────────────────────────────────────────────
CLASS_NAMES = [
    "apple__apple_scab", "apple__black_rot", "apple__cedar_apple_rust",
    "apple__healthy", "cassava__bacterial_blight_cbb",
    "cassava__brown_streak_disease_cbsd", "cassava__green_mottle_cgm",
    "cassava__healthy", "cassava__mosaic_disease_cmd",
    "cherry_including_sour__healthy", "cherry_including_sour__powdery_mildew",
    "corn_maize__cercospora_leaf_spot_gray_leaf_spot", "corn_maize__common_rust",
    "corn_maize__healthy", "corn_maize__northern_leaf_blight",
    "grape__black_rot", "grape__esca_black_measles", "grape__healthy",
    "grape__leaf_blight_isariopsis_leaf_spot", "orange__haunglongbing_citrus_greening",
    "peach__bacterial_spot", "peach__healthy", "pepper_bell__bacterial_spot",
    "pepper_bell__healthy", "potato__early_blight", "potato__healthy",
    "potato__late_blight", "rice__brownspot", "rice__healthy", "rice__hispa",
    "rice__leafblast", "squash__powdery_mildew", "strawberry__healthy",
    "strawberry__leaf_scorch", "tomato__bacterial_spot", "tomato__early_blight",
    "tomato__healthy", "tomato__late_blight", "tomato__leaf_mold",
    "tomato__septoria_leaf_spot", "tomato__spider_mites_two-spotted_spider_mite",
    "tomato__target_spot", "tomato__tomato_mosaic_virus",
    "tomato__tomato_yellow_leaf_curl_virus",
]

IMG_SIZE = (224, 224)

def format_class_name(raw):
    parts = raw.split("__")
    crop    = parts[0].replace("_", " ").title()
    disease = parts[1].replace("_", " ").replace("-", " ").title() if len(parts) > 1 else ""
    return crop, disease

def preprocess_image(img):
    img = img.resize(IMG_SIZE)
    img_array = np.array(img, dtype=np.float32)
    # EfficientNet preprocess_input manually (same as training)
    img_array /= 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_array = (img_array - mean) / std
    return np.expand_dims(img_array, axis=0)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "ok", "model": "EfficientNetB2-TFLite", "classes": len(CLASS_NAMES)})

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    file = request.files["image"]
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400
    try:
        start = time.time()

        interpreter = get_interpreter()
        input_details  = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_input = preprocess_image(img)

        interpreter.set_tensor(input_details[0]["index"], img_input)
        interpreter.invoke()
        preds = interpreter.get_tensor(output_details[0]["index"])[0]

        elapsed  = round((time.time() - start) * 1000)
        top_idx  = int(np.argmax(preds))
        top_conf = float(preds[top_idx])
        crop, disease = format_class_name(CLASS_NAMES[top_idx])

        top5_indices = np.argsort(preds)[::-1][:5]
        top5 = []
        for i in top5_indices:
            c, d = format_class_name(CLASS_NAMES[i])
            top5.append({"raw": CLASS_NAMES[i], "crop": c, "disease": d, "confidence": float(preds[i])})

        is_healthy = "healthy" in CLASS_NAMES[top_idx].lower()
        severity = "None" if is_healthy else ("High" if top_conf >= 0.85 else ("Medium" if top_conf >= 0.55 else "Low"))

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
