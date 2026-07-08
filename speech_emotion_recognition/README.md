# 🎙️ Speech Emotion Recognition (SER)

Recognize human emotions (happy, angry, sad, neutral, fearful, disgust, surprised, calm)
from speech audio using deep learning + speech signal processing.

**Approach:** MFCC feature extraction → CNN + LSTM (CRNN) model → Flask web app for
uploading/recording audio and getting a live emotion prediction.

Datasets supported out-of-the-box: **RAVDESS**, **TESS**, **EMO-DB (Berlin)**.

---

## 📁 File Structure

```
speech_emotion_recognition/
├── app.py                     # Flask backend (serves frontend + prediction API)
├── requirements.txt           # Python dependencies
├── README.md
│
├── model/
│   ├── train_model.py         # Trains the CNN-LSTM model on MFCC features
│   ├── predict.py             # Loads trained model, predicts emotion for a wav file
│   ├── emotion_model.h5       # (generated after training) saved Keras model
│   └── label_encoder.pkl      # (generated after training) label encoder
│
├── utils/
│   └── feature_extraction.py  # MFCC + audio feature extraction utilities
│
├── dataset/                   # Put RAVDESS / TESS / EMO-DB audio files here
│   └── (Actor_01/, Actor_02/, ... or TESS folders, or EMO-DB wav/)
│
├── uploads/                   # Temp storage for audio uploaded via the web app
│
├── templates/
│   └── index.html             # Frontend UI
│
└── static/
    ├── css/style.css          # Styling (dark theme)
    └── js/script.js           # Upload / record audio, call API, render result
```

---

## ⚙️ Setup

```bash
cd speech_emotion_recognition
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 🧠 Step 1 — Get a dataset

Download one (or more) of:
- **RAVDESS**: https://zenodo.org/record/1188976 (Audio_Speech_Actors_01-24.zip)
- **TESS**: https://tspace.library.utoronto.ca/handle/1807/24487
- **EMO-DB**: http://emodb.bilderbar.info/download/

Extract into `dataset/`, e.g.:
```
dataset/
├── Actor_01/*.wav
├── Actor_02/*.wav
...
```
(TESS/EMO-DB also work — the label parser auto-detects the naming convention.)

## 🏋️ Step 2 — Train the model

```bash
python model/train_model.py --data_dir dataset --epochs 50
```
This extracts MFCC features from every clip, trains a CNN-LSTM classifier,
and saves `model/emotion_model.h5` + `model/label_encoder.pkl`.

## 🚀 Step 3 — Run the web app

```bash
python app.py
```
Open **http://127.0.0.1:5000** — upload a `.wav`/`.mp3` file or record directly
from your mic, and see the predicted emotion with confidence scores.

## 🔍 Predict from command line (optional)

```bash
python model/predict.py --file path/to/audio.wav
```

---

## 🧩 Model Architecture

```
Input (MFCC: 40 x 174)
   │
Conv1D(128) → BatchNorm → ReLU → MaxPool → Dropout
   │
Conv1D(256) → BatchNorm → ReLU → MaxPool → Dropout
   │
LSTM(128) → Dropout
   │
LSTM(64)
   │
Dense(64) → ReLU → Dropout
   │
Dense(num_emotions) → Softmax
```

CNN layers learn local spectral patterns in the MFCCs; the LSTM layers learn
temporal dynamics across frames (how emotion evolves through the utterance).

## 🎯 Emotions Recognized

`neutral, calm, happy, sad, angry, fearful, disgust, surprised`
(subset depends on which dataset(s) you train on).
