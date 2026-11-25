"""Correct the checked-class frequency bias before blending the two course models."""
from course_input import course_data
import json

import numpy as np
import pandas as pd
from scipy.special import softmax

from compare_course_ensembles import (REFERENCE, ROOT, credit, log_probs, metrics,
                                      paired_interval, ranks, temperature_power)

OUT = ROOT / "results/native_upgrade/final_course"


def pool(parts, powers, weight, exponent, counts):
    git, previous = [power*log_probs(p) for p, power in zip(parts, powers)]
    previous -= exponent * np.log(counts/counts.mean())
    return (1-weight)*softmax(git, axis=1) + weight*softmax(previous, axis=1)


def main():
    train = pd.read_csv(course_data(ROOT) / 'train.csv', dtype={"id": str})
    checked = train[train.checked == 1]
    classes = np.sort(train.label.unique())
    y = np.searchsorted(classes, checked.label)
    counts = np.bincount(y)
    by_seed = {}
    for seed, filename in [(73, "selection_oof"), (419, "confirmation_oof")]:
        with (np.load(REFERENCE / f"{filename}.npz") as git,
              np.load(ROOT / f"results/course_comparison/previous_seed{seed}.npz") as old):
            if not (np.array_equal(old["ids"], checked.id) and np.array_equal(git["y"], y)
                    and np.array_equal(old["y"], y)):
                raise ValueError("Course fold predictions are misaligned")
            by_seed[seed] = (git["pred"], old["pred"])
    powers = [temperature_power(p, y) for p in by_seed[73]]
    grid = []
    # All classes have either 10 or 40 checked examples. Each of the five stratified
    # folds therefore has the same class proportions as the full checked set.
    for w in [0, .125, .25, .375, .5]:
        for exponent in [0, .5, 1]:
            result = metrics(pool(by_seed[73], powers, w, exponent, counts), y)
            grid.append(dict(weight=w, exponent=exponent, **result))
    best = max(grid, key=lambda r: r["macro_map3"] + r["map3"])
    w, exp = best["weight"], best["exponent"]
    val = pool(by_seed[419], powers, w, exp, counts)
    delta = credit(val, y) - credit(by_seed[419][0], y)
    report = dict(powers=powers, weight=w, exponent=exp, selection=best,
                  confirmation=metrics(val, y), reference=metrics(by_seed[419][0], y),
                  improved=int((delta>0).sum()), worsened=int((delta<0).sum()),
                  paired_ci=paired_interval(delta, y), selection_grid=grid)
    sample = pd.read_csv(course_data(ROOT) / 'sample_submission.csv', dtype=str)
    parts = []
    for path in [REFERENCE / "test_predictions.npz", ROOT / "test_predictions.npz"]:
        with np.load(path, allow_pickle=True) as f:
            if not np.array_equal(f["classes"], classes) or not np.array_equal(f["ids"], sample.id):
                raise ValueError(f"Invalid test order in {path}")
            parts.append(f["probabilities"])
    scores = pool(parts, powers, w, exp, counts)
    sample["label"] = [" ".join(row) for row in classes[ranks(scores)]]
    OUT.mkdir(parents=True, exist_ok=True)
    diagnostic = pd.DataFrame({"id":checked.id.to_numpy(), "label":classes[y],
                               "first_prediction":classes[ranks(val)[:, 0]],
                               "repository_credit":credit(by_seed[419][0], y),
                               "blend_credit":credit(val, y)})
    diagnostic.to_csv(OUT / "checked_predictions.csv", index=False)
    sample.to_csv(OUT / "submission.csv", index=False)
    if len(sample) != 1592 or not pd.read_csv(OUT/"submission.csv",dtype=str).equals(sample):
        raise ValueError("Invalid course submission")
    (OUT / "metrics.json").write_text(json.dumps(report, indent=2))
    print({k:v for k,v in report.items() if k != "selection_grid"})


if __name__ == "__main__":
    main()
