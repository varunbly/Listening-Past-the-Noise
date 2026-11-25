"""Combine the published course model with a separately trained acoustic model.

Use checked-label fold predictions from seed 73 for selection and from seed
419 for confirmation. No Freesound metadata or leaderboard labels are inputs.
"""
from course_input import course_data
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp, softmax
from sklearn.model_selection import StratifiedKFold


ROOT = Path(__file__).resolve().parent
REFERENCE = ROOT.parent / "reference0836" / "best_model_0836" / "round4"
UPGRADE = ROOT / "results" / "native_upgrade" / "course"


def ranks(scores):
    return np.argsort(-scores, axis=1, kind="stable")[:, :3]


def credit(scores, y):
    hits = ranks(scores) == y[:, None]
    return (hits / np.arange(1, 4)).sum(axis=1)


def metrics(scores, y):
    rr = credit(scores, y)
    return dict(map3=rr.mean(), macro_map3=np.mean([
        rr[y == c].mean() for c in np.unique(y)
    ]), top1=(rr == 1).mean(), top3=(rr > 0).mean())


def log_probs(p):
    p = np.asarray(p, dtype=float)
    if not np.isfinite(p).all() or (p < 0).any() or (p.sum(axis=1) <= 0).any():
        raise ValueError("Invalid probability matrix")
    return np.log(np.maximum(p / p.sum(axis=1, keepdims=True), 1e-10))


def temperature_power(p, y):
    # Differences in confidence should not determine the mixture weights.
    logs = log_probs(p)
    rows = np.arange(len(y))

    def loss(power):
        scores = logs * power
        return np.mean(logsumexp(scores, axis=1) - scores[rows, y])

    fit = minimize_scalar(loss, bounds=(0.15, 6), method="bounded")
    if not fit.success:
        raise ValueError("Temperature fit failed")
    return float(fit.x)


def combine(parts, powers, w):
    first, second = [softmax(log_probs(p) * power, axis=1)
                     for p, power in zip(parts, powers)]
    return (1 - w) * first + w * second


def sorted_predictions(path, expected_ids, classes, fold=None):
    with np.load(path, allow_pickle=False) as data:
        key = "ids" if "ids" in data else "names"
        ids = pd.Index([Path(s).stem for s in data[key].astype(str)])
        columns = pd.Index(data["classes"].astype(str))
        if not ids.is_unique or set(ids) != set(expected_ids):
            raise ValueError(f"Audio ID mismatch in {path}")
        if not columns.is_unique or set(columns) != set(classes):
            raise ValueError(f"Class mismatch in {path}")
        row_index = ids.get_indexer(expected_ids)
        pred = data["probabilities"][row_index][:, columns.get_indexer(classes)]
        if pred.shape != (len(expected_ids), len(classes)):
            raise ValueError(f"Invalid shape in {path}")
        if fold is not None:
            observed = data["fold"][row_index]
            # Verify the partition, allowing either zero- or one-based fold names.
            for i in np.unique(fold):
                if not np.array_equal(fold == i, observed == observed[fold == i][0]):
                    raise ValueError(f"Fold assignment mismatch in {path}")
    return pred


def paired_interval(delta, y):
    # Resample the same clips for both models within each class, preserving counts.
    rng = np.random.default_rng(271)
    totals = np.zeros(3000)
    for c in np.unique(y):
        block = delta[y == c]
        totals += rng.choice(block, size=(len(totals), len(block))).sum(axis=1)
    return np.quantile(totals / len(y), [.025, .975]).tolist()


def main(reference, upgrade, output):
    data = course_data(ROOT)
    train = pd.read_csv(data / "train.csv", dtype={"id": str})
    template = pd.read_csv(data / "sample_submission.csv", dtype=str)
    report = json.loads((reference / "final_report.json").read_text())
    for name in ("train", "test"):
        digest = hashlib.sha256((data / f"{name}.csv").read_bytes()).hexdigest()
        if digest != report[f"{name}_csv_sha256"]:
            raise ValueError(f"Course {name} split differs from the reference")
    classes = np.sort(train.label.unique())
    checked = train.loc[train.checked == 1]
    ids = checked.id.to_numpy()
    y = np.searchsorted(classes, checked.label)
    splits = {}
    for seed, ref_file, new_file in ((73, "selection_oof.npz", "oof73.npz"),
                                     (419, "confirmation_oof.npz", "oof.npz")):
        fold = np.empty(len(y), dtype=int)
        for i, (_, val) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(ids, y)):
            fold[val] = i
        with np.load(reference / ref_file) as old:
            if not np.array_equal(y, old["y"]):
                raise ValueError("Reference checked-label order differs")
            p = old["pred"]
        q = sorted_predictions(upgrade / new_file, ids, classes, fold)
        splits[seed] = (p, q, fold)
    selection = splits[73][:2]
    powers = [temperature_power(p, y) for p in selection]
    grid = {}
    for w in [0, .25, .5, .75, 1]:
        grid[w] = metrics(combine(selection, powers, w), y)
    # Give equal importance to the competition metric and its class-balanced analogue.
    weight = max(grid, key=lambda w: grid[w]["map3"] + grid[w]["macro_map3"])
    old, new, fold = splits[419]
    mixed = combine([old, new], powers, weight)
    a, b = credit(old, y), credit(mixed, y)
    reference_result = metrics(old, y)
    candidate_result = metrics(mixed, y)
    confirmed = bool(weight > 0 and b.mean() > a.mean()
                     and candidate_result["macro_map3"] >= reference_result["macro_map3"])
    confirmation = dict(reference=reference_result, acoustic_model=metrics(new, y),
                        candidate=candidate_result, map3_gain=float((b-a).mean()),
                        paired_percentile_interval=paired_interval(b-a, y),
                        improved_clips=int((b>a).sum()), worsened_clips=int((b<a).sum()),
                        fold_gains=[float((b-a)[fold == i].mean()) for i in range(5)])
    test_ids = template.id.to_numpy()
    if not np.array_equal(test_ids, pd.read_csv(data / "test.csv", dtype=str).id):
        raise ValueError("Test template order differs")
    old_test = sorted_predictions(reference / "test_predictions.npz", test_ids, classes)
    new_test = sorted_predictions(upgrade / "test.npz", test_ids, classes)
    test_scores = combine([old_test, new_test], powers, weight)
    test_top = ranks(test_scores)
    output.mkdir(parents=True, exist_ok=True)
    labels = [" ".join(row) for row in classes[test_top]]
    template.assign(label=labels).to_csv(output / "submission_project4.csv", index=False)
    saved = pd.read_csv(output / "submission_project4.csv", dtype=str)
    if (not saved.id.is_unique or not np.array_equal(saved.id, test_ids)
            or labels != saved.label.tolist()):
        raise ValueError("Submission failed round-trip check")
    result = dict(selection_seed=73, confirmation_seed=419, checked_clips=len(y),
                  selection=grid, acoustic_weight=weight, temperature_powers=powers,
                  confirmation=confirmation, confirmed_vs_reference=confirmed,
                  test_rows=len(test_ids), test_first_choice_changes=int(
                      (test_top[:, 0] != ranks(old_test)[:, 0]).sum()))
    (output / "comparison.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    parser.add_argument("--upgrade", type=Path, default=UPGRADE)
    parser.add_argument("--output", type=Path, default=UPGRADE.parent / "course_comparison")
    args = parser.parse_args()
    main(args.reference, args.upgrade, args.output)
