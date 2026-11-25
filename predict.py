"""Fit the fixed Project 4 components from the supplied audio and write ranked predictions."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
LOCAL = HERE / "group13"
GROUP = HERE / "reference0836/best_model_0836"


def final_predictions(data, output):
    import numpy as np
    import pandas as pd

    sys.path.insert(0, str(LOCAL))
    from course_top3 import arbitrate, base_choices, rank_lower
    from compare_course_ensembles import ranks
    from select_course_graph import matrices

    train = pd.read_csv(data / "train.csv", dtype={"id": str})
    test = pd.read_csv(data / "test.csv", dtype=str)
    with np.load(HERE / "ranker_weights.npz") as fitted:
        classes = fitted["classes"]
        checked = train.loc[train.checked.eq(1), "label"]
        if not np.array_equal(classes, np.sort(checked.unique())):
            raise ValueError("Course class codes differ from the fitted classifier")
        counts = checked.value_counts().reindex(classes).to_numpy()
        baseline, features = matrices("test", test.id.to_numpy(), counts)
        lower = rank_lower(baseline, features, fitted["lower"])
        scores = arbitrate(lower, baseline, features, fitted["arbitration"], base_choices("test"))
    if not test.id.is_unique or not np.isfinite(scores).all() or scores.shape != (len(test),len(classes)):
        raise ValueError("Invalid prediction matrix")
    test.assign(label=[" ".join(row) for row in classes[ranks(scores)]]).to_csv(output, index=False)
    print(f"Saved {len(test)} ranked predictions to {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Extracted course dataset directory")
    parser.add_argument("--output", type=Path, default=Path("submission.csv"))
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()
    data = args.data.resolve(strict=True)
    for child in ("train.csv","test.csv","audio/audio/train","audio/audio/test"):
        if not (data/child).exists():
            parser.error(f"Missing course input: {data/child}")
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    env = os.environ.copy()
    env["PROJECT4_DATA"] = str(data)
    jobs = str(args.jobs)
    steps = [
        (GROUP, ["run_best.py", "--retrain", "--jobs", jobs]),
        (LOCAL, ["improve.py", "--stage", "fisher", "--jobs", jobs]),
        (LOCAL, ["deliver.py", "--skip-holdout", "--jobs", jobs]),
        (LOCAL, ["native_features.py", "course", "--workers", jobs]),
        (LOCAL, ["course_local_graph.py"]),
        (LOCAL, ["course_noise_mixture.py"]),
        (LOCAL, ["train_course_graph.py"]),
        (LOCAL, ["course_pair_classifier.py"]),
        (LOCAL, ["course_envelope.py"]),
        (LOCAL, ["course_segment_ridge.py"]),
        (LOCAL, ["course_forest_transfer.py"]),
    ]
    for folder, command in steps:
        subprocess.run([sys.executable, "-u", *command], cwd=folder, env=env, check=True)
    final_predictions(data, args.output.resolve())


if __name__ == "__main__":
    main()
