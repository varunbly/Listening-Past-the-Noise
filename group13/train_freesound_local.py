"""Fit classical audio classifiers on the two local Freesound archives using matched course labels."""
from course_input import course_data
from pathlib import Path
import json
import zipfile

import joblib
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

BASE = Path(__file__).resolve().parent
CACHE = BASE / "results" / "archive_comparison"
OUTPUT = BASE / "results" / "freesound_course_labels"


def read_archive(stem):
    with zipfile.ZipFile(BASE.parent / (stem + ".zip")) as archive:
        wavs = [item for item in archive.infolist() if item.filename.lower().endswith(".wav")]
    manifest_path = CACHE / (stem + "_manifest.npz")
    if not manifest_path.is_file():
        manifest_path = CACHE / (stem + ".npz")
    with np.load(manifest_path) as manifest:
        if ([item.filename for item in wavs] != manifest["names"].tolist()
                or [item.CRC for item in wavs] != manifest["crc"].tolist()
                or [item.file_size for item in wavs] != manifest["size"].tolist()):
            raise ValueError(f"Audio archive differs from the comparison cache: {stem}")
        metadata = pd.DataFrame({k: manifest[k] for k in ("names", "n", "sr", "crc", "size")})
    metadata["duration"] = metadata.n / metadata.sr
    return metadata


def spectral_features(stem, metadata):
    # The comparison cache stores measurements from each original WAV independently.
    with np.load(CACHE / (stem + "_spec.npz")) as data:
        quantiles = data["q"].reshape(len(metadata), -1)
        bands = data["s"].astype("float32")
    windows = bands.reshape(len(bands), 4, 24, 48)
    return np.column_stack((quantiles, windows.mean(2).reshape(len(bands), -1),
                            windows.std(2).reshape(len(bands), -1),
                            np.abs(np.diff(bands, axis=1)).mean(1),
                            np.log1p(metadata.duration))).astype("float32")


def matched_labels(sources):
    course = read_archive("project4")
    pairs = np.load(CACHE / "confirmed_pairs.npy")
    correlation, index, speed, lag = pairs.T
    index = index.astype(int)
    duration = sources.duration.to_numpy()[index] / speed
    delay = lag / 2205
    overlap = (np.minimum(course.duration, duration + delay) - np.maximum(0, delay))
    keep = (index >= 0) & (correlation >= .80) & (overlap >= .90 * np.minimum(course.duration, duration))
    chosen = course.loc[keep, ["names"]].copy()
    chosen["source"] = index[keep]
    # Exclude sources linked to the assignment test split or multiple course IDs.
    one = chosen.loc[~chosen.source.duplicated(keep=False) & chosen.names.str.contains("/train/")].copy()
    one["id"] = one.names.map(lambda name: Path(name).stem)
    labels = pd.read_csv(course_data(BASE) / "train.csv", dtype={"id": str})
    return one.merge(labels, on="id", validate="one_to_one").set_index("source")


def fit_ensemble(x, labels, weights):
    svm = make_pipeline(StandardScaler(), SVC(C=10, class_weight="balanced"))
    forest = ExtraTreesClassifier(n_estimators=400, class_weight="balanced", min_samples_leaf=2,
                                  random_state=42, n_jobs=4)
    svm.fit(x, labels, svc__sample_weight=weights)
    forest.fit(x, labels, sample_weight=weights)
    if not np.array_equal(svm.classes_, forest.classes_):
        raise ValueError("Class order differs between models")
    return svm, forest


def predict_top_three(svm, forest, test_x):
    scores = .5 * softmax(svm.decision_function(test_x), axis=1) + .5 * forest.predict_proba(test_x)
    return svm.classes_[np.argsort(-scores, axis=1)[:, :3]]


def main():
    stems = ["FSDKaggle2018.audio_train", "FSDKaggle2018.audio_test"]
    train, test = (read_archive(stem) for stem in stems)
    sources = pd.concat([train, test], ignore_index=True)
    labels = matched_labels(sources)
    train_rows = labels.loc[labels.index < len(train)].copy()
    test_rows = labels.loc[(labels.index >= len(train)) & labels.checked.eq(1)].copy()
    # Keep exact duplicates out of training if the test archive also contains that WAV.
    keys = pd.MultiIndex.from_frame(sources[["crc", "size"]])
    train_rows = train_rows.loc[~keys[train_rows.index].isin(keys[len(train):])]
    if train_rows.empty or test_rows.empty:
        raise ValueError("Need labeled train examples and checked test examples")
    train_x, test_x = (spectral_features(s, m) for s, m in zip(stems, (train, test)))
    x = train_x[train_rows.index]
    weights = np.where(train_rows.checked.eq(1), 1., .4)
    svm, forest = fit_ensemble(x, train_rows.label, weights)
    top3 = predict_top_three(svm, forest, test_x)
    ranks = top3[test_rows.index.to_numpy() - len(train)] == test_rows.label.to_numpy()[:, None]
    report = dict(train_archive_wavs=len(train), train_labeled=len(train_rows),
                  train_checked=int(train_rows.checked.sum()), train_classes=len(svm.classes_),
                  test_predictions=len(test), test_checked_scored=len(test_rows),
                  test_classes_scored=test_rows.label.nunique(), top1_correct=int(ranks[:,0].sum()),
                  correct_by_rank=ranks.sum(0).tolist(), map3=float((ranks / [1, 2, 3]).sum(1).mean()),
                  top1_accuracy=float(ranks[:, 0].mean()), top3_recall=float(ranks.any(1).mean()),
                  label_source="Project 4 training CSV, transferred by audio similarity. Only checked test matches scored.")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"fname": test.names.map(lambda s: Path(s).name),
                  "label": [" ".join(row) for row in top3]}).to_csv(OUTPUT / "predictions.csv", index=False)
    joblib.dump((svm, forest), OUTPUT / "models.joblib", compress=1)
    (OUTPUT / "metrics.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
