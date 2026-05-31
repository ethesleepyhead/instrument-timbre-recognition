"""
Shared feature extraction functions and constants for the instrument timbre project.

Used by both train_model.py (training) and timbre_api.py (inference).
"""

import librosa
import numpy as np

SAMPLE_RATE = 22050

# ---------------------------------------------------------------------------
# Label display names — canonical mapping for all 11 instrument codes
# ---------------------------------------------------------------------------
LABEL_DISPLAY_NAMES = {
    "cel": "Cello",
    "cla": "Clarinet",
    "flu": "Flute",
    "gac": "Acoustic Guitar",
    "gel": "Electric Guitar",
    "org": "Organ",
    "pia": "Piano",
    "sax": "Saxophone",
    "tru": "Trumpet",
    "vio": "Violin",
    "voi": "Voice",
}

# ---------------------------------------------------------------------------
# Instrument colors — for timeline rendering (11-color palette, stable order)
# ---------------------------------------------------------------------------
INSTRUMENT_COLORS = [
    "#4f46e5", "#0d9488", "#d97706", "#dc2626",
    "#7c3aed", "#059669", "#db2777", "#2563eb",
    "#ea580c", "#65a30d", "#0891b2",
]

# ---------------------------------------------------------------------------
# Timeline / sliding-window defaults
# ---------------------------------------------------------------------------
WINDOW_SEC = 1.0        # seconds per analysis window
HOP_SEC = 0.5            # seconds between window starts (50% overlap)
TOP_K = 2                # top-k predictions kept per window
MIN_CONFIDENCE = 0.10    # drop predictions below this threshold


# ---------------------------------------------------------------------------
# In-memory feature extraction (y, sr) → feature vector
# ---------------------------------------------------------------------------

def _extract_features_40_from_audio(y, sr):
    """MFCC-only: 20 mean + 20 std = 40 features — core logic on audio array."""
    if len(y) == 0:
        return None
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    return np.concatenate([
        np.mean(mfcc, axis=1),
        np.std(mfcc, axis=1),
    ]).astype(np.float32)


def _extract_features_76_from_audio(y, sr):
    """v3: single STFT, no hpss/tempo — 76 features — core logic on audio array."""
    if len(y) == 0:
        return None

    S = np.abs(librosa.stft(y))
    S_power = S ** 2

    mel = librosa.feature.melspectrogram(S=S_power, sr=sr)
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel), sr=sr, n_mfcc=20)

    spectral_centroid = librosa.feature.spectral_centroid(S=S, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr)
    chroma = librosa.feature.chroma_stft(S=S, sr=sr)

    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y)

    spectral_energy = float(np.mean(S))
    spectral_energy_std = float(np.std(S))

    feature_blocks = [
        np.mean(mfcc, axis=1),              # 20
        np.std(mfcc, axis=1),               # 20
        np.mean(chroma, axis=1),            # 12
        np.std(chroma, axis=1),             # 12
        np.array([
            np.mean(spectral_centroid),
            np.std(spectral_centroid),
            np.mean(spectral_bandwidth),
            np.std(spectral_bandwidth),
            np.mean(spectral_rolloff),
            np.std(spectral_rolloff),
            np.mean(zero_crossing_rate),
            np.std(zero_crossing_rate),
            np.mean(rms),
            np.std(rms),
            spectral_energy,
            spectral_energy_std,
        ]),
    ]
    return np.concatenate(feature_blocks).astype(np.float32)


def _extract_features_77_from_audio(y, sr):
    """v1/v2: original extraction with hpss + tempo — 77 features — core logic."""
    if len(y) == 0:
        return None

    harmonic, percussive = librosa.effects.hpss(y)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    rms = librosa.feature.rms(y=y)
    tempo, _ = librosa.beat.beat_track(y=percussive, sr=sr)

    feature_blocks = [
        np.mean(mfcc, axis=1),
        np.std(mfcc, axis=1),
        np.mean(chroma, axis=1),
        np.std(chroma, axis=1),
        np.array([
            np.mean(spectral_centroid),
            np.std(spectral_centroid),
            np.mean(spectral_bandwidth),
            np.std(spectral_bandwidth),
            np.mean(spectral_rolloff),
            np.std(spectral_rolloff),
            np.mean(zero_crossing_rate),
            np.std(zero_crossing_rate),
            np.mean(rms),
            np.std(rms),
            float(tempo),
            np.mean(np.abs(harmonic)),
            np.mean(np.abs(percussive)),
        ]),
    ]
    return np.concatenate(feature_blocks).astype(np.float32)


# ---------------------------------------------------------------------------
# File-based feature extraction (wraps the in-memory functions above)
# ---------------------------------------------------------------------------

def extract_features_40(file_path):
    """MFCC-only: 20 mean + 20 std = 40 features."""
    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
        return _extract_features_40_from_audio(y, sr)
    except Exception as exc:
        print(f"Feature extraction error (40d): {exc}")
        return None


def extract_features_76(file_path):
    """v3: single STFT, no hpss/tempo — 76 features."""
    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
        return _extract_features_76_from_audio(y, sr)
    except Exception as exc:
        print(f"Feature extraction error (76d): {exc}")
        return None


def extract_features_77(file_path):
    """v1/v2: original extraction with hpss + tempo — 77 features."""
    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
        return _extract_features_77_from_audio(y, sr)
    except Exception as exc:
        print(f"Feature extraction error (77d): {exc}")
        return None


# n_features → file-based extraction function
EXTRACTORS = {
    40: extract_features_40,
    76: extract_features_76,
    77: extract_features_77,
}

# n_features → in-memory extraction function  (y, sr) → feature vector
EXTRACTORS_FROM_AUDIO = {
    40: _extract_features_40_from_audio,
    76: _extract_features_76_from_audio,
    77: _extract_features_77_from_audio,
}


def get_extractor(n_features):
    """Return the file-based feature extraction function for the given dimension."""
    fn = EXTRACTORS.get(n_features)
    if fn is None:
        raise ValueError(
            f"Model expects {n_features} features, but no matching extractor "
            f"is registered. Available dimensions: {sorted(EXTRACTORS.keys())}"
        )
    return fn


def get_extractor_from_audio(n_features):
    """Return the in-memory feature extraction function for the given dimension."""
    fn = EXTRACTORS_FROM_AUDIO.get(n_features)
    if fn is None:
        raise ValueError(
            f"Model expects {n_features} features, but no matching in-memory extractor "
            f"is registered. Available dimensions: {sorted(EXTRACTORS_FROM_AUDIO.keys())}"
        )
    return fn
