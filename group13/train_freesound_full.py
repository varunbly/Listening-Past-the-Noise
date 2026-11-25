"""Train and score the full FSDKaggle2018 release with its native labels."""
import argparse
import json
from pathlib import Path
from zipfile import ZipFile

import joblib
import numpy as np
import pandas as pd

from train_freesound_local import BASE, fit_ensemble, predict_top_three, read_archive, spectral_features


def ordered_labels(metadata, member, recordings):
    with ZipFile(metadata) as archive, archive.open("FSDKaggle2018.meta/" + member) as stream:
        labels = pd.read_csv(stream).set_index("fname", verify_integrity=True)
    names = pd.Index([Path(s).name for s in recordings.names])
    if not names.is_unique or set(names) != set(labels.index) or labels.label.isna().any():
        raise ValueError(f"Labels must cover each archive WAV exactly once: {member}")
    return labels.loc[names]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=Path.home() / "Downloads" / "FSDKaggle2018.meta.zip")
    args = parser.parse_args()
    train_stem, test_stem = "FSDKaggle2018.audio_train", "FSDKaggle2018.audio_test"
    train, test = read_archive(train_stem), read_archive(test_stem)
    labels = ordered_labels(args.metadata, "train_post_competition.csv", train)
    if len(train) != 9473 or len(test) != 1600 or not labels.manually_verified.isin([0, 1]).all():
        raise ValueError("Unexpected Freesound release or training verification flags")
    weights = np.where(labels.manually_verified.eq(1), 1., .4)
    print(f"Fitting SVM and Extra Trees on all {len(train)} labeled WAVs", flush=True)
    svm, forest = fit_ensemble(spectral_features(train_stem, train), labels.label, weights)
    predicted = predict_top_three(svm, forest, spectral_features(test_stem, test))
    out = BASE / "results" / "freesound_full"
    out.mkdir(exist_ok=True, parents=True)
    # Native category names can contain punctuation. Preserve each rank as its own field.
    predictions = pd.DataFrame(predicted, columns=["rank1", "rank2", "rank3"])
    predictions.insert(0, "fname", [Path(name).name for name in test.names])
    predictions.to_csv(out / "predictions.csv", index=False)
    joblib.dump((svm, forest), out / "models.joblib", compress=1)
    truth = ordered_labels(args.metadata, "test_post_competition_scoring_clips.csv", test)
    hits = predicted == truth.label.to_numpy()[:, None]
    report = dict(train_wavs=len(labels), train_verified=int(labels.manually_verified.sum()),
                  classes=len(svm.classes_), test_wavs=len(test), test_scored=len(truth),
                  correct_by_rank=hits.sum(axis=0).tolist(), map3=float((hits / [1, 2, 3]).sum(1).mean()),
                  top1_accuracy=float(hits[:, 0].mean()), top3_recall=float(hits.any(axis=1).mean()),
                  exact_archive_overlap=int(pd.MultiIndex.from_frame(test[["crc","size"]]).isin(
                      pd.MultiIndex.from_frame(train[["crc","size"]])).sum()))
    (out / "metrics.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
