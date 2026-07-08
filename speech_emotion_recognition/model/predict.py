"""
predict.py
----------
Loads the trained CNN-LSTM model and predicts the emotion of a given
audio file. Can be used from the command line or imported by app.py.

CLI usage:
    python model/predict.py --file path/to/audio.wav
"""

import os
import sys
import argparse
import pickle
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.feature_extraction import extract_features, load_audio

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(MODEL_DIR, "emotion_model.h5")
ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")

EMOTION_EMOJI = {
    "neutral": "😐", "calm": "😌", "happy": "😄", "sad": "😢",
    "angry": "😠", "fearful": "😨", "disgust": "🤢", "surprised": "😲",
    "boredom": "🥱",
}

_model = None
_label_encoder = None


def _lazy_load():
    """Load the Keras model + label encoder only once, on first use."""
    global _model, _label_encoder
    if _model is None:
        from tensorflow.keras.models import load_model
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"No trained model found at {MODEL_PATH}. "
                "Run `python model/train_model.py` first."
            )
        _model = load_model(MODEL_PATH)
        with open(ENCODER_PATH, "rb") as f:
            _label_encoder = pickle.load(f)
    return _model, _label_encoder


def predict_emotion(file_path):
    """
    Predict the emotion for a given audio file.
    Returns a dict: {emotion, emoji, confidence, all_scores}
    """
    model, le = _lazy_load()

    audio, sr = load_audio(file_path)
    features = extract_features(audio=audio, sr=sr)     # (channels, time)
    features = np.transpose(features, (1, 0))            # (time, channels)
    features = np.expand_dims(features, axis=0)           # (1, time, channels)

    probs = model.predict(features, verbose=0)[0]
    predicted_idx = int(np.argmax(probs))
    predicted_label = le.inverse_transform([predicted_idx])[0]

    all_scores = {
        le.inverse_transform([i])[0]: float(probs[i])
        for i in range(len(probs))
    }
    all_scores = dict(sorted(all_scores.items(), key=lambda x: x[1], reverse=True))

    return {
        "emotion": predicted_label,
        "emoji": EMOTION_EMOJI.get(predicted_label, "🎙️"),
        "confidence": float(probs[predicted_idx]),
        "all_scores": all_scores,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict emotion from a speech audio file")
    parser.add_argument("--file", type=str, required=True, help="Path to .wav/.mp3 file")
    args = parser.parse_args()

    result = predict_emotion(args.file)
    print(f"\nPredicted Emotion: {result['emotion'].upper()} {result['emoji']}")
    print(f"Confidence: {result['confidence'] * 100:.2f}%\n")
    print("All scores:")
    for emotion, score in result["all_scores"].items():
        print(f"  {emotion:12s}: {score * 100:5.2f}%")
