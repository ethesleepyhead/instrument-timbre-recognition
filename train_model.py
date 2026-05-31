import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from features import (
    LABEL_DISPLAY_NAMES,
    SAMPLE_RATE,
    extract_features_76,
)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset")
MODEL_PATH = os.path.join(BASE_DIR, "timbre_model.pkl")
CACHE_DIR = os.path.join(BASE_DIR, "features_cache")
MIN_SAMPLES_PER_LABEL = 5


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def discover_labels(dataset_path):
    labels = []
    for entry in sorted(os.listdir(dataset_path)):
        full = os.path.join(dataset_path, entry)
        if os.path.isdir(full):
            wav_count = sum(
                1 for n in os.listdir(full) if n.lower().endswith(".wav")
            )
            if wav_count >= MIN_SAMPLES_PER_LABEL:
                labels.append(entry)
    return labels


def get_dataset_hash(dataset_path, labels):
    """Deterministic hash over label names + file names + file sizes.

    Changing / adding / removing any audio file invalidates the cache.
    """
    hasher = hashlib.md5()
    for label in sorted(labels):
        hasher.update(label.encode())
        folder = os.path.join(dataset_path, label)
        for name in sorted(os.listdir(folder)):
            if name.lower().endswith(".wav"):
                full = os.path.join(folder, name)
                hasher.update(name.encode())
                hasher.update(str(os.path.getsize(full)).encode())
    return hasher.hexdigest()[:12]


# ---------------------------------------------------------------------------
# Progress helpers
# ---------------------------------------------------------------------------

def _progress_bar(current, total, label="", start_time=None, width=30):
    fraction = min(current / total, 1.0) if total > 0 else 0
    filled = int(width * fraction)
    bar = "#" * filled + "-" * (width - filled)
    parts = [f"  [{bar}] {current}/{total}  {label}"]
    if start_time and current > 0:
        elapsed = time.time() - start_time
        rate = current / elapsed if elapsed > 0 else 0
        eta = (total - current) / rate if rate > 0 else 0
        parts.append(f" | {elapsed:.0f}s elapsed | ~{eta:.0f}s remaining")
    sys.stdout.write("\r" + "".join(parts))
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Feature loading with caching
# ---------------------------------------------------------------------------

def load_or_extract_features(dataset_path, labels, force=False):
    """Return (X, y).  Load from cache when possible; otherwise extract + cache."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_key = get_dataset_hash(dataset_path, labels)
    feat_path = os.path.join(CACHE_DIR, f"features_{cache_key}.npy")
    lab_path = os.path.join(CACHE_DIR, f"labels_{cache_key}.npy")

    if not force and os.path.exists(feat_path) and os.path.exists(lab_path):
        print(f"\n[1/4] Loading cached features  (key: {cache_key})")
        X = np.load(feat_path)
        y = np.load(lab_path)
        print(f"      {len(X)} feature vectors loaded from cache.\n")
        return X, y

    print(f"\n[1/4] Extracting audio features  (key: {cache_key})")

    # Count files for progress
    total_files = 0
    class_file_counts = {}
    for label in labels:
        folder = os.path.join(dataset_path, label)
        names = [n for n in os.listdir(folder) if n.lower().endswith(".wav")]
        class_file_counts[label] = len(names)
        total_files += len(names)

    print(f"      {total_files} files across {len(labels)} classes")
    for label in labels:
        print(f"        {label}: {class_file_counts[label]} files")

    features_list = []
    target_list = []
    processed = 0
    t0 = time.time()

    for label in labels:
        folder = os.path.join(dataset_path, label)
        file_names = sorted(
            [n for n in os.listdir(folder) if n.lower().endswith(".wav")]
        )
        for fname in file_names:
            feats = extract_features_76(os.path.join(folder, fname))
            if feats is not None:
                features_list.append(feats)
                target_list.append(label)
            processed += 1
            if processed % 20 == 0 or processed == total_files:
                _progress_bar(processed, total_files, label, t0)

    print()  # final newline
    elapsed = time.time() - t0
    print(f"      Done in {elapsed:.1f}s  "
          f"({len(features_list)} valid / {processed} files)\n")

    X = np.array(features_list)
    y = np.array(target_list)

    np.save(feat_path, X)
    np.save(lab_path, y)
    print(f"      Features cached → {CACHE_DIR}\n")
    return X, y


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(n_estimators=600):
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=None,
                    min_samples_leaf=1,
                    random_state=42,
                    n_jobs=-1,
                    class_weight="balanced_subsample",
                ),
            ),
        ]
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Train instrument timbre classifier (Random Forest)."
    )
    p.add_argument(
        "--quick", action="store_true",
        help="Quick dev mode: fewer trees, fewer CV folds, faster iteration.",
    )
    p.add_argument(
        "--force-extract", action="store_true",
        help="Ignore feature cache and re-extract from audio files.",
    )
    p.add_argument(
        "--no-cache", action="store_true",
        help="Disable feature caching entirely.",
    )
    p.add_argument(
        "--labels", type=str, default=None,
        help="Comma-separated instrument codes to include (default: all in dataset/).",
    )
    args = p.parse_args()

    # --- Mode banner --------------------------------------------------------
    if args.quick:
        n_estimators = 100
        cv_folds = 3
        print("=" * 58)
        print("  QUICK MODE  (--quick)")
        print(f"  n_estimators={n_estimators}   cv_folds={cv_folds}")
        print("=" * 58)
    else:
        n_estimators = 600
        cv_folds = 5
        print("=" * 58)
        print("  FULL TRAINING MODE")
        print(f"  n_estimators={n_estimators}   cv_folds={cv_folds}")
        print("=" * 58)

    # --- Discover labels ----------------------------------------------------
    if not os.path.isdir(DATASET_PATH):
        raise FileNotFoundError(f"Dataset directory not found: {DATASET_PATH}")

    labels = discover_labels(DATASET_PATH)

    # Filter by --labels if provided
    if args.labels:
        requested = [s.strip() for s in args.labels.split(",") if s.strip()]
        filtered = [l for l in labels if l in requested]
        missing = [l for l in requested if l not in labels]
        if missing:
            print(f"Warning: these requested labels were not found: {', '.join(missing)}")
        labels = filtered

    if len(labels) < 2:
        raise ValueError("Need at least 2 label folders with audio samples.")

    print(f"\nDetected {len(labels)} labels: {', '.join(labels)}")

    # --- Phase 1 — features -------------------------------------------------
    x, y = load_or_extract_features(
        DATASET_PATH, labels,
        force=args.force_extract or args.no_cache,
    )

    if len(x) == 0:
        raise ValueError("No usable audio features extracted.")

    unique, counts = np.unique(y, return_counts=True)
    sample_counts = dict(zip(unique, map(int, counts)))
    print(f"Sample counts: {json.dumps(sample_counts, ensure_ascii=False)}")
    print(f"Total samples: {len(x)}   feature dim: {x.shape[1]}\n")

    # --- Phase 2 — split ----------------------------------------------------
    print("[2/4] Train / test split")
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y,
    )
    print(f"      train: {len(x_train)}   test: {len(x_test)}\n")

    # --- Phase 3 — cross-validation -----------------------------------------
    print(f"[3/4] Cross-validation  ({cv_folds}-fold)")

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_scores = []
    t_cv = time.time()

    for fold, (tr_idx, va_idx) in enumerate(cv.split(x_train, y_train), 1):
        t_fold = time.time()
        fm = build_model(n_estimators=n_estimators)
        fm.fit(x_train[tr_idx], y_train[tr_idx])
        score = fm.score(x_train[va_idx], y_train[va_idx])
        cv_scores.append(score)
        print(f"      fold {fold}/{cv_folds}  accuracy={score:.4f}  "
              f"({time.time() - t_fold:.1f}s)")

    cv_scores = np.array(cv_scores)
    print(f"      CV accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}"
          f"  (total {time.time() - t_cv:.1f}s)\n")

    # --- Phase 4 — final training + evaluation ------------------------------
    print("[4/4] Final training + evaluation")
    model = build_model(n_estimators=n_estimators)

    t_train = time.time()
    model.fit(x_train, y_train)
    print(f"      Training: {time.time() - t_train:.1f}s")

    accuracy = model.score(x_test, y_test)
    print(f"      Test accuracy: {accuracy:.4f}\n")

    predictions = model.predict(x_test)
    report = classification_report(
        y_test, predictions, output_dict=True, zero_division=0,
    )

    print("Per-class report:")
    for label in sorted(labels):
        if label in report:
            m = report[label]
            print(f"  {label:>6s}  "
                  f"prec={m['precision']:.3f}  "
                  f"rec={m['recall']:.3f}  "
                  f"f1={m['f1-score']:.3f}  "
                  f"sup={int(m['support'])}")

    # --- Save ---------------------------------------------------------------
    artifact = {
        "model": model,
        "labels": labels,
        "sample_rate": SAMPLE_RATE,
        "feature_version": 3,
        "feature_dim": int(x.shape[1]),
        "label_display_names": {
            lbl: LABEL_DISPLAY_NAMES.get(lbl, lbl)
            for lbl in labels
        },
        "metrics": {
            "test_accuracy": float(accuracy),
            "cv_accuracy_mean": float(cv_scores.mean()),
            "cv_accuracy_std": float(cv_scores.std()),
            "sample_counts": sample_counts,
            "report": report,
            "train_distribution": dict(Counter(y_train)),
            "test_distribution": dict(Counter(y_test)),
        },
    }
    joblib.dump(artifact, MODEL_PATH)
    print(f"\nModel saved → {MODEL_PATH}")
    print("Done.")


if __name__ == "__main__":
    main()
