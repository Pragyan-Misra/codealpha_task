"""
train_model.py
--------------
Trains a CNN-LSTM (CRNN) model for Speech Emotion Recognition using
MFCC-based features extracted from RAVDESS / TESS / EMO-DB audio clips.

Usage:
    python model/train_model.py --data_dir dataset --epochs 50 --batch_size 32
"""

import os
import sys
import argparse
import pickle
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.feature_extraction import (
    extract_features, load_audio, add_noise, pitch_shift,
    parse_label_from_filename, get_feature_shape
)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv1D, BatchNormalization, Activation, MaxPooling1D, Dropout,
    LSTM, Dense
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam


def collect_dataset(data_dir, augment=True):
    """Walk the dataset directory, extract features + labels for every wav file."""
    X, y = [], []
    wav_files = []
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.lower().endswith((".wav", ".mp3")):
                wav_files.append(os.path.join(root, f))

    if not wav_files:
        raise FileNotFoundError(
            f"No .wav/.mp3 files found under '{data_dir}'. "
            "Download RAVDESS/TESS/EMO-DB and place them there."
        )

    print(f"Found {len(wav_files)} audio files. Extracting features...")
    for path in tqdm(wav_files):
        label = parse_label_from_filename(os.path.basename(path))
        if label is None:
            continue  # skip files we can't label

        try:
            audio, sr = load_audio(path)
        except Exception as e:
            print(f"Skipping {path}: {e}")
            continue

        # Original sample
        X.append(extract_features(audio=audio, sr=sr))
        y.append(label)

        # Simple augmentation to boost dataset size / robustness
        if augment:
            try:
                X.append(extract_features(audio=add_noise(audio), sr=sr))
                y.append(label)
                X.append(extract_features(audio=pitch_shift(audio, sr, n_steps=2), sr=sr))
                y.append(label)
            except Exception:
                pass

    return np.array(X), np.array(y)


def build_model(input_shape, num_classes):
    model = Sequential([
        Conv1D(128, kernel_size=5, padding="same", input_shape=input_shape),
        BatchNormalization(),
        Activation("relu"),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),

        Conv1D(256, kernel_size=5, padding="same"),
        BatchNormalization(),
        Activation("relu"),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),

        LSTM(128, return_sequences=True),
        Dropout(0.3),
        LSTM(64),

        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    parser = argparse.ArgumentParser(description="Train Speech Emotion Recognition model")
    parser.add_argument("--data_dir", type=str, default="dataset", help="Path to dataset folder")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--no_augment", action="store_true", help="Disable data augmentation")
    args = parser.parse_args()

    model_dir = os.path.dirname(os.path.abspath(__file__))

    # 1. Feature extraction
    X, y = collect_dataset(args.data_dir, augment=not args.no_augment)
    print(f"Dataset shape: X={X.shape}, y={y.shape}")

    # Reshape: (samples, time_steps, features) for Conv1D/LSTM -> transpose feature axis
    X = np.transpose(X, (0, 2, 1))  # (samples, time, feature_channels)

    # 2. Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    y_categorical = to_categorical(y_encoded)
    print("Classes:", list(le.classes_))

    # 3. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_categorical, test_size=0.2, random_state=42, stratify=y_categorical
    )

    # 4. Build + train model
    model = build_model(input_shape=(X.shape[1], X.shape[2]), num_classes=y_categorical.shape[1])
    model.summary()

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6),
        ModelCheckpoint(
            os.path.join(model_dir, "emotion_model.h5"),
            monitor="val_accuracy", save_best_only=True
        ),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
    )

    # 5. Save label encoder + final model
    with open(os.path.join(model_dir, "label_encoder.pkl"), "wb") as f:
        pickle.dump(le, f)
    model.save(os.path.join(model_dir, "emotion_model.h5"))

    # 6. Final evaluation
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nFinal Test Accuracy: {acc * 100:.2f}%  |  Loss: {loss:.4f}")
    print(f"Model saved to: {os.path.join(model_dir, 'emotion_model.h5')}")
    print(f"Label encoder saved to: {os.path.join(model_dir, 'label_encoder.pkl')}")


if __name__ == "__main__":
    main()
