# Instrument Timbre Recognition Web Demo

A lightweight audio ML project for instrument classification and timeline visualization.

Upload an audio file, run inference with a trained scikit-learn model, and view:

- whole-clip class probabilities
- top predicted instrument
- sliding-window instrument activity timeline

## What this project does

This project combines:

- audio feature extraction with `librosa`
- classical machine learning with `scikit-learn`
- a local inference API with `Flask`
- a browser demo UI with waveform playback and timeline rendering

It is designed as a compact, explainable ML demo rather than a production-grade music analysis system.

## Features

- Upload audio files: `WAV`, `MP3`, `FLAC`, `OGG`, `M4A`
- Play uploaded audio with waveform display
- Predict instrument probabilities for the full clip
- Visualize top prediction and per-class confidence bars
- Run sliding-window analysis for temporal instrument activity
- Render a per-instrument timeline under the waveform

## Project structure

```text
audio-app/
  index.html          # Frontend page structure
  script.js           # Frontend behavior and API integration
  style.css           # Frontend styling
  features.py         # Shared feature extraction logic and constants
  timbre_api.py       # Flask app, static serving, inference endpoints
  train_model.py      # Model training script
  timbre_model.pkl    # Trained model artifact
  run.bat             # Windows startup script
  dataset/            # Training audio grouped by label
  features_cache/     # Cached extracted features for training
  README.md
```

## Architecture

### 1. Frontend

Files:

- [index.html](C:/Users/30328/Desktop/audio-app/index.html)
- [script.js](C:/Users/30328/Desktop/audio-app/script.js)
- [style.css](C:/Users/30328/Desktop/audio-app/style.css)

Responsibilities:

- file upload and drag-drop
- waveform playback with WaveSurfer
- calling backend endpoints
- rendering full-clip probabilities
- rendering the instrument timeline

Main flow:

1. User uploads an audio file.
2. Frontend sends the file to `/analyze` and `/analyze_timeline`.
3. Frontend renders:
   - overall prediction summary
   - probability bars
   - timeline rows and segments

### 2. Backend API

File:

- [timbre_api.py](C:/Users/30328/Desktop/audio-app/timbre_api.py)

Responsibilities:

- serve the frontend at `http://127.0.0.1:5000`
- load the trained model from `timbre_model.pkl`
- validate uploaded files
- run full-clip inference
- run sliding-window timeline inference
- return structured JSON for the frontend

Routes:

- `GET /` : frontend page
- `GET /health` : model/config status
- `POST /analyze` : full-clip prediction
- `POST /analyze_timeline` : timeline prediction

### 3. Feature extraction

File:

- [features.py](C:/Users/30328/Desktop/audio-app/features.py)

Responsibilities:

- define reusable feature extraction functions
- keep training-time and inference-time features consistent
- expose shared constants such as:
  - label display names
  - color palette
  - timeline window parameters

Current feature sets supported by the codebase:

- `40` dimensions: MFCC mean + std
- `76` dimensions: MFCC + chroma + spectral features
- `77` dimensions: legacy variant with extra tempo/HPSS-related values

The backend selects the correct extractor based on the actual trained model input dimension.

### 4. Training pipeline

File:

- [train_model.py](C:/Users/30328/Desktop/audio-app/train_model.py)

Responsibilities:

- discover labels from `dataset/`
- extract and cache features
- split train/test data
- run cross-validation
- train `StandardScaler + RandomForestClassifier`
- save the final artifact to `timbre_model.pkl`

Training modes:

- quick mode: `python train_model.py --quick`
- full mode: `python train_model.py`
- force feature re-extraction: `python train_model.py --force-extract`

## Inference flow

### Full-clip inference

`POST /analyze`

Flow:

1. save uploaded file temporarily
2. extract features
3. run model prediction
4. return sorted class probabilities

Example response:

```json
{
  "results": [
    {"label": "pia", "confidence": 0.82},
    {"label": "vio", "confidence": 0.11}
  ],
  "top_prediction": {"label": "pia", "confidence": 0.82},
  "labels": ["cla", "flu", "gac", "pia", "tru", "vio"],
  "model_info": {
    "n_features": 76,
    "test_accuracy": 0.70
  }
}
```

### Timeline inference

`POST /analyze_timeline`

Flow:

1. load full audio
2. split audio into sliding windows
3. extract features per window
4. predict top-k labels for each window
5. merge adjacent compatible segments
6. return timeline-ready JSON

Default timeline parameters are defined in `features.py`.

Example response:

```json
{
  "segments": [
    {
      "start": 0.0,
      "end": 1.5,
      "predictions": [
        {"label": "pia", "confidence": 0.81},
        {"label": "vio", "confidence": 0.22}
      ]
    }
  ],
  "labels": ["cla", "flu", "gac", "pia", "tru", "vio"],
  "duration": 3.0,
  "window_size": 1.0,
  "hop_size": 0.5
}
```

## Run locally

### Option 1

```bash
python timbre_api.py
```

### Option 2

Double-click:

```text
run.bat
```

Then open:

```text
http://127.0.0.1:5000
```

## Dependencies

Install the core dependencies with:

```bash
pip install flask flask-cors joblib librosa numpy scikit-learn
```

## Screenshots

Suggested location for screenshots:

```text
github/screenshots/
```

Placeholder paths:

- `github/screenshots/home.png`
- `github/screenshots/result.png`
- `github/screenshots/timeline.png`
- `github/screenshots/training.png`

## Notes

- The current system is best understood as a dominant-instrument / approximate multi-candidate demo.
- Timeline output is based on sliding-window prediction, not true frame-level multi-label source analysis.
- Performance depends heavily on dataset coverage, label quality, and recording conditions.

