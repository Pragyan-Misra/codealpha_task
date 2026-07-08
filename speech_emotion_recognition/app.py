"""
app.py
------
Flask backend for the Speech Emotion Recognition web app.

Routes:
    GET  /            -> Frontend UI
    POST /predict      -> Accepts an audio file, returns predicted emotion JSON
    GET  /health        -> Simple health check (model loaded or not)
"""

import os
import uuid
import traceback
from flask import Flask, request, jsonify, render_template

from model.predict import predict_emotion, MODEL_PATH

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
ALLOWED_EXTENSIONS = {"wav", "mp3", "webm", "ogg", "m4a"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB max upload


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "model_trained": os.path.exists(MODEL_PATH),
    })


@app.route("/predict", methods=["POST"])
def predict():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided. Use form field name 'audio'."}), 400

    file = request.files["audio"]
    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": f"Unsupported file type. Allowed: {ALLOWED_EXTENSIONS}"}), 400

    ext = file.filename.rsplit(".", 1)[1].lower()
    temp_filename = f"{uuid.uuid4().hex}.{ext}"
    temp_path = os.path.join(app.config["UPLOAD_FOLDER"], temp_filename)
    file.save(temp_path)

    try:
        result = predict_emotion(temp_path)
        return jsonify(result)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Failed to process audio: {e}"}), 500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
