"""
feature_extraction.py
----------------------
Speech signal processing utilities for Speech Emotion Recognition.

Extracts MFCCs (Mel-Frequency Cepstral Coefficients) plus a few
complementary features (delta-MFCC, zero-crossing rate, RMS energy)
from an audio waveform, and pads/truncates them to a fixed length so
they can be fed into a CNN-LSTM network.
"""

import numpy as np
import librosa

SAMPLE_RATE = 22050
N_MFCC = 40
MAX_PAD_LEN = 174          # ~4 seconds of audio at default hop length
DURATION = 4.0             # seconds - clips are trimmed/padded to this length


def load_audio(file_path, sr=SAMPLE_RATE, duration=DURATION):
    """Load an audio file, resample, and fix its length."""
    audio, sample_rate = librosa.load(file_path, sr=sr, duration=duration)
    target_len = int(sr * duration)
    if len(audio) < target_len:
        audio = np.pad(audio, (0, target_len - len(audio)))
    else:
        audio = audio[:target_len]
    return audio, sample_rate


def add_noise(audio, noise_factor=0.005):
    """Data augmentation: inject light Gaussian noise."""
    noise = np.random.randn(len(audio))
    return audio + noise_factor * noise


def pitch_shift(audio, sr, n_steps=2):
    """Data augmentation: shift pitch up/down slightly."""
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)


def stretch(audio, rate=1.1):
    """Data augmentation: time-stretch the clip."""
    return librosa.effects.time_stretch(audio, rate=rate)


def extract_features(file_path=None, audio=None, sr=SAMPLE_RATE,
                      n_mfcc=N_MFCC, max_pad_len=MAX_PAD_LEN):
    """
    Extract a (n_mfcc, max_pad_len) feature matrix from an audio file or
    an already-loaded waveform.

    Combines:
      - MFCCs                (timbre / vocal tract shape)
      - Delta-MFCCs          (rate of change -> prosody/emotion cues)
      - Zero-crossing rate   (voiced/unvoiced, harshness)
      - RMS energy           (loudness envelope -> arousal)
    """
    if audio is None:
        audio, sr = load_audio(file_path, sr=sr)

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    delta_mfcc = librosa.feature.delta(mfcc)
    zcr = librosa.feature.zero_crossing_rate(audio)
    rms = librosa.feature.rms(y=audio)

    # Stack all features along the "channel" (frequency) axis
    features = np.vstack([mfcc, delta_mfcc, zcr, rms])

    # Pad or truncate along the time axis to a fixed length
    if features.shape[1] < max_pad_len:
        pad_width = max_pad_len - features.shape[1]
        features = np.pad(features, pad_width=((0, 0), (0, pad_width)), mode="constant")
    else:
        features = features[:, :max_pad_len]

    return features.astype(np.float32)


def get_feature_shape():
    """Total feature rows = n_mfcc + delta_mfcc + zcr(1) + rms(1)."""
    return (N_MFCC * 2 + 2, MAX_PAD_LEN)


# ---------------------------------------------------------------------
# Dataset label parsing (RAVDESS / TESS / EMO-DB naming conventions)
# ---------------------------------------------------------------------

RAVDESS_EMOTION_MAP = {
    "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
    "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised",
}

EMODB_EMOTION_MAP = {
    "W": "angry", "L": "boredom", "E": "disgust", "A": "fearful",
    "F": "happy", "T": "sad", "N": "neutral",
}

TESS_KEYWORDS = ["angry", "disgust", "fear", "happy", "neutral", "ps", "sad"]


def parse_label_from_filename(filename):
    """
    Auto-detect dataset naming convention and return the emotion label.
    Supports RAVDESS, TESS, and EMO-DB (Berlin) filenames.
    """
    name = filename.lower()
    base = filename.split("/")[-1].split("\\")[-1]

    # RAVDESS: 03-01-05-01-02-01-12.wav  -> 3rd field is emotion code
    parts = base.split("-")
    if len(parts) >= 3 and parts[2] in RAVDESS_EMOTION_MAP:
        return RAVDESS_EMOTION_MAP[parts[2]]

    # TESS: OAF_back_angry.wav / YAF_date_happy.wav
    for kw in TESS_KEYWORDS:
        if kw in name:
            return "fearful" if kw == "fear" else ("surprised" if kw == "ps" else kw)

    # EMO-DB: 03a01Wa.wav -> 6th character is the emotion code letter
    stem = base.split(".")[0]
    if len(stem) >= 6:
        code = stem[5].upper()
        if code in EMODB_EMOTION_MAP:
            return EMODB_EMOTION_MAP[code]

    return None
