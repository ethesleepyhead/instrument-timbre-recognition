import os
import tempfile

import joblib
import librosa
import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from features import (
    INSTRUMENT_COLORS,
    HOP_SEC,
    LABEL_DISPLAY_NAMES,
    MIN_CONFIDENCE,
    SAMPLE_RATE,
    TOP_K,
    WINDOW_SEC,
    get_extractor,
    get_extractor_from_audio,
)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "timbre_model.pkl")
ALLOWED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

app = Flask(__name__)
CORS(app)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _get_model_input_dim(model):
    """Read n_features_in_ from the RF classifier, handling both Pipeline and raw."""
    if hasattr(model, "named_steps"):
        # Pipeline — find the classifier step
        for name in ("classifier", "rf", "randomforestclassifier"):
            step = model.named_steps.get(name)
            if step is not None and hasattr(step, "n_features_in_"):
                return step.n_features_in_
    if hasattr(model, "n_features_in_"):
        return model.n_features_in_
    raise ValueError("Cannot determine expected feature count from model.")


def load_model_artifact():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "Model file not found. Run  python train_model.py --quick  to train a model first."
        )

    artifact = joblib.load(MODEL_PATH)

    if isinstance(artifact, dict) and "model" in artifact:
        model = artifact["model"]
        labels = artifact.get("labels", list(getattr(model, "classes_", [])))
        metrics = artifact.get("metrics", {})
        meta = {
            "feature_version": artifact.get("feature_version"),
            "feature_dim": artifact.get("feature_dim"),
        }
    else:
        model = artifact
        labels = list(getattr(model, "classes_", []))
        metrics = {}
        meta = {}

    if not labels:
        raise ValueError("No labels found in the trained model.")

    n_features = _get_model_input_dim(model)
    return model, labels, metrics, n_features, meta


def is_allowed_file(filename):
    _, ext = os.path.splitext(filename.lower())
    return ext in ALLOWED_EXTENSIONS


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")


@app.get("/script.js")
def script():
    return send_from_directory(BASE_DIR, "script.js")


@app.get("/style.css")
def style():
    return send_from_directory(BASE_DIR, "style.css")


@app.get("/health")
def health():
    try:
        _, labels, metrics, n_features, _meta = load_model_artifact()
        label_display = {
            lbl: LABEL_DISPLAY_NAMES.get(lbl, lbl)
            for lbl in labels
        }
        colors = {
            lbl: INSTRUMENT_COLORS[i % len(INSTRUMENT_COLORS)]
            for i, lbl in enumerate(labels)
        }
        return jsonify({
            "status": "ok",
            "labels": labels,
            "n_features": n_features,
            "test_accuracy": metrics.get("test_accuracy"),
            "label_display_names": label_display,
            "instrument_colors": colors,
            "window_params": {
                "window_sec": WINDOW_SEC,
                "hop_sec": HOP_SEC,
                "top_k": TOP_K,
                "min_confidence": MIN_CONFIDENCE,
            },
        })
    except Exception as exc:
        return jsonify({"status": "error", "error": str(exc)}), 500


@app.post("/analyze")
def analyze():
    if "file" not in request.files:
        return jsonify({"error": "No file provided."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "The uploaded file has no name."}), 400
    if not is_allowed_file(file.filename):
        return jsonify({
            "error": "Unsupported file type. Supported: WAV, MP3, FLAC, OGG, M4A."
        }), 400

    temp_path = None

    try:
        model, labels, metrics, n_features, _meta = load_model_artifact()
        extract_fn = get_extractor(n_features)

        suffix = os.path.splitext(file.filename)[1] or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            file.save(tmp.name)
            temp_path = tmp.name

        features = extract_fn(temp_path)
        if features is None:
            return jsonify({
                "error": "Unable to extract audio features. The file may be corrupted or silent."
            }), 422

        features = features.reshape(1, -1)
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(features)[0]
            model_labels = list(getattr(model, "classes_", labels))
            results = [
                {"label": label, "confidence": float(prob)}
                for label, prob in zip(model_labels, probabilities)
            ]
        else:
            prediction = model.predict(features)[0]
            results = [
                {"label": label, "confidence": 1.0 if label == prediction else 0.0}
                for label in labels
            ]

        results.sort(key=lambda item: item["confidence"], reverse=True)
        top_result = results[0] if results else None

        label_display = {
            lbl: LABEL_DISPLAY_NAMES.get(lbl, lbl)
            for lbl in model_labels
        }
        return jsonify({
            "results": results,
            "top_prediction": top_result,
            "labels": labels,
            "label_display_names": label_display,
            "model_info": {
                "n_features": n_features,
                "test_accuracy": metrics.get("test_accuracy"),
            },
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


# ---------------------------------------------------------------------------
# Timeline analysis  —  sliding-window prediction
# ---------------------------------------------------------------------------


def _extract_segment_features(y_segment, sr, n_features):
    """Extract features from an in-memory audio segment — no temp files."""
    extract_fn = get_extractor_from_audio(n_features)
    return extract_fn(y_segment, sr)


def _merge_segments(windows, total_duration, hop_sec):
    """Merge consecutive windows sharing the same top-1 label.

    Uses running weighted-average confidence (not naive pairwise mean) and
    bridges gaps of ≤1 hop between segments with the same top-1 label.
    """
    if not windows:
        return []

    # Pass 1: merge consecutive same-label windows with weighted confidence
    merged = []
    for win in windows:
        if not win["predictions"]:
            continue

        top_label = win["predictions"][0]["label"]

        if merged and merged[-1]["predictions"][0]["label"] == top_label:
            merged[-1]["end"] = win["end"]
            n = merged[-1]["_count"] + 1
            merged[-1]["_count"] = n
            # Running weighted average: old = old * (n-1)/n + new / n
            for p_new in win["predictions"]:
                for p_old in merged[-1]["predictions"]:
                    if p_old["label"] == p_new["label"]:
                        p_old["confidence"] = round(
                            p_old["confidence"] * (n - 1) / n + p_new["confidence"] / n, 4
                        )
                        break
                else:
                    if len(merged[-1]["predictions"]) < TOP_K:
                        merged[-1]["predictions"].append(p_new)
            merged[-1]["predictions"].sort(key=lambda x: x["confidence"], reverse=True)
        else:
            merged.append({
                "start": win["start"],
                "end": win["end"],
                "predictions": [dict(p) for p in win["predictions"]],
                "_count": 1,
            })

    # Pass 2: bridge gaps ≤1 hop between segments with the same top-1 label
    bridged = []
    for seg in merged:
        if bridged and bridged[-1]["predictions"][0]["label"] == seg["predictions"][0]["label"]:
            gap = seg["start"] - bridged[-1]["end"]
            if gap <= hop_sec + 0.01:
                bridged[-1]["end"] = seg["end"]
                n1, n2 = bridged[-1]["_count"], seg["_count"]
                total_w = n1 + n2
                for p_new in seg["predictions"]:
                    for p_old in bridged[-1]["predictions"]:
                        if p_old["label"] == p_new["label"]:
                            p_old["confidence"] = round(
                                (p_old["confidence"] * n1 + p_new["confidence"] * n2) / total_w, 4
                            )
                            break
                    else:
                        if len(bridged[-1]["predictions"]) < TOP_K:
                            bridged[-1]["predictions"].append(p_new)
                bridged[-1]["predictions"].sort(key=lambda x: x["confidence"], reverse=True)
                bridged[-1]["_count"] = total_w
                continue
        bridged.append(seg)

    # Clean up internal bookkeeping
    for seg in bridged:
        seg.pop("_count", None)
        seg["end"] = min(seg["end"], total_duration)

    return bridged


@app.post("/analyze_timeline")
def analyze_timeline():
    if "file" not in request.files:
        return jsonify({"error": "No file provided."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "The uploaded file has no name."}), 400
    if not is_allowed_file(file.filename):
        return jsonify({
            "error": "Unsupported file type. Supported: WAV, MP3, FLAC, OGG, M4A."
        }), 400

    temp_path = None

    try:
        model, labels, metrics, n_features, _meta = load_model_artifact()

        # Save uploaded file and load full audio
        suffix = os.path.splitext(file.filename)[1] or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            file.save(tmp.name)
            temp_path = tmp.name

        y, sr = librosa.load(temp_path, sr=SAMPLE_RATE, mono=True)
        if len(y) == 0:
            return jsonify({"error": "Audio file is empty or silent."}), 422

        total_duration = len(y) / sr

        # Build windows
        window_samples = int(WINDOW_SEC * sr)
        hop_samples = int(HOP_SEC * sr)

        # Floor: at least one window covering the whole audio
        if len(y) < window_samples:
            windows_starts = [0]
            window_samples = len(y)
        else:
            windows_starts = list(range(0, len(y) - window_samples + 1, hop_samples))

        raw_windows = []
        for start_idx in windows_starts:
            end_idx = start_idx + window_samples
            segment = y[start_idx:end_idx]
            start_sec = round(start_idx / sr, 2)

            feats = _extract_segment_features(segment, sr, n_features)
            if feats is None:
                continue

            feats = feats.reshape(1, -1)
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(feats)[0]
                model_labels = list(getattr(model, "classes_", labels))
            else:
                probs = np.zeros(len(labels))
                pred = model.predict(feats)[0]
                probs[labels.index(pred) if pred in labels else 0] = 1.0
                model_labels = labels

            # Collect top-k above threshold
            items = [
                {"label": lbl, "confidence": round(float(p), 4)}
                for lbl, p in zip(model_labels, probs)
                if p >= MIN_CONFIDENCE
            ]
            items.sort(key=lambda x: x["confidence"], reverse=True)
            items = items[:TOP_K]

            raw_windows.append({
                "start": start_sec,
                "end": round(min(start_sec + WINDOW_SEC, total_duration), 2),
                "predictions": items,
            })

        segments = _merge_segments(raw_windows, total_duration, HOP_SEC)

        label_display = {
            lbl: LABEL_DISPLAY_NAMES.get(lbl, lbl)
            for lbl in labels
        }
        return jsonify({
            "segments": segments,
            "labels": labels,
            "label_display_names": label_display,
            "duration": round(total_duration, 2),
            "window_size": WINDOW_SEC,
            "hop_size": HOP_SEC,
            "model_info": {
                "n_features": n_features,
                "test_accuracy": metrics.get("test_accuracy"),
            },
        })

    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import webbrowser

    HOST = "127.0.0.1"
    PORT = 5000
    URL = f"http://{HOST}:{PORT}"

    print()
    print("=" * 56)
    print("  乐器音色识别 — ML Demo Server")
    print(f"  浏览器访问  {URL}")
    print("=" * 56)
    print()

    if os.path.exists(MODEL_PATH):
        try:
            _, labels, metrics, nf, meta = load_model_artifact()
            acc = metrics.get("test_accuracy")
            acc_s = f"{float(acc)*100:.1f}%" if acc is not None else "N/A"
            fv = meta.get("feature_version", "?")
            fd = meta.get("feature_dim", nf)
            display_names = [LABEL_DISPLAY_NAMES.get(l, l) for l in labels]
            print(f"  Model        : {len(labels)} classes, {nf}d features (v{fv}, dim={fd}), accuracy ~{acc_s}")
            print(f"  Labels       : {', '.join(labels)}")
            print(f"  Display      : {', '.join(display_names)}")
        except Exception as e:
            print(f"  Warning: model file exists but failed to load: {e}")
    else:
        print("  Warning: timbre_model.pkl not found.")
        print("  Run  python train_model.py --quick  to train first.")

    print()
    webbrowser.open(URL)
    app.run(host=HOST, port=PORT, debug=False, use_reloader=False)
