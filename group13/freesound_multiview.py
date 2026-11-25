"""Fit and validate a classical multi-view ensemble on the native Freesound splits."""
import argparse
from pathlib import Path

import numpy as np
from scipy import linalg, sparse
from scipy.special import softmax
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, normalize

from train_freesound_local import BASE, read_archive
from train_freesound_full import ordered_labels

FEATURES = BASE / "results" / "freesound_features"
OUT = BASE / "results" / "freesound_improved"
META = Path.home() / "Downloads" / "FSDKaggle2018.meta.zip"
VIEWS = ("fisher", "patches", "texture", "tonal", "covariance", "combined", "fusion")


def load_training():
    audio = read_archive("FSDKaggle2018.audio_train")
    labels = ordered_labels(META, "train_post_competition.csv", audio)
    classes, y = np.unique(labels.label.to_numpy(dtype=str), return_inverse=True)
    checked = labels.manually_verified.eq(1).to_numpy()
    ids = np.flatnonzero(checked)
    _, val = train_test_split(ids, test_size=.2, stratify=y[ids], random_state=2026)
    return audio, classes, y, checked, np.sort(val)


def map3(probabilities, y):
    ranks = np.argsort(-probabilities, axis=1, kind="stable")[:, :3]
    return float(((ranks == y[:, None]) / [1, 2, 3]).sum(axis=1).mean())


def kernel(x, other):
    # Mix two RBF bandwidths on unit-normalized descriptors.
    out = x @ other.T
    out = np.minimum(out, 1.)
    out -= 1.
    return .4 * np.exp(2 * out) + .6 * np.exp(5 * out)


def solve_ridge(gram, targets, weights, regularization):
    system = gram.copy(order="F")
    system.flat[::len(system)+1] += regularization / weights
    return linalg.solve(system, targets, assume_a="pos", overwrite_a=True, check_finite=False)


def neighbors(gram, k=14, self_edges=False):
    rows, cols, vals = [], [], []
    for start in range(0, len(gram), 192):
        block = gram[start:start+192].copy()
        if self_edges:
            ind = np.arange(len(block))
            block[ind, start+ind] = -1
        idx = np.argpartition(-block, k-1, axis=1)[:, :k]
        similarity = np.take_along_axis(block, idx, axis=1).clip(1.e-8)
        rows.append(np.repeat(np.arange(start, start+len(block)), k))
        cols.append(idx.ravel())
        vals.append((similarity**2).ravel())
    return sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=gram.shape)


def predict_view(x, query, y, checked, n_classes, scale, with_svm=False):
    if scale:
        scaler = StandardScaler().fit(x)
        x = scaler.transform(x).clip(-5., 5.)
        query = scaler.transform(query).clip(-5., 5.)
    x = normalize(x)
    query = normalize(query)
    gram = kernel(x, x)
    cross = kernel(query, x)
    targets = np.eye(n_classes, dtype="float32")[y]
    clean = np.flatnonzero(checked)
    noisy = np.flatnonzero(~checked)
    pilot = solve_ridge(gram[np.ix_(clean, clean)], targets[clean], np.ones(len(clean)), .4)
    hints = gram[np.ix_(noisy, clean)] @ pilot
    rank = (hints > hints[np.arange(len(noisy)), y[noisy], None]).sum(1)
    weights = np.ones(len(x), dtype="float32")
    weights[noisy] = np.select([rank == 0, rank < 3, rank < 8], [.85, .65, .40], default=.2)
    coefficients = solve_ridge(gram, targets, weights, .40)
    decision = cross @ coefficients
    ridge = softmax(10 * decision, axis=1)
    graph = neighbors(gram, self_edges=True)
    transition = normalize((graph + graph.T).tocsr(), norm="l1")
    mixing = np.where(checked, .1, 1 - .65 * weights).astype("float32")[:,None]
    source = (1-mixing) * targets
    belief = targets.copy()
    for _ in range(45):
        updated = source + mixing*(transition @ belief)
        if np.max(abs(updated-belief)) < 1.e-5:
            break
        belief = updated
    evidence = normalize(neighbors(cross), norm="l1") @ belief
    prior = targets[clean].mean(0).clip(1.e-5)
    graph_scores = normalize(evidence / prior**.25, norm="l1")
    result = {"ridge": ridge, "graph": graph_scores}
    if with_svm:
        model = SVC(C=9., kernel="precomputed", decision_function_shape="ovr", cache_size=192)
        model.fit(gram, y, sample_weight=weights)
        result["svm"] = softmax(model.decision_function(cross) / 1.6, axis=1)
    return result


def split_features(view, fit, val, audio, test):
    with np.load(FEATURES / (view+".npz"), allow_pickle=False) as data:
        x = data["train"].astype("float32", copy=False)
        if "train_names" in data:
            assert [Path(s).name for s in data["train_names"].astype(str)] == [Path(s).name for s in audio.names]
        if len(x) != len(audio) or not np.isfinite(x).all():
            raise ValueError(f"Invalid training features for {view}")
        if test is None:
            return x[fit], x[val]
        query = data["test"].astype("float32", copy=False)
        if "test_names" in data:
            assert [Path(s).name for s in data["test_names"].astype(str)] == [Path(s).name for s in test.names]
        assert len(query) == len(test) and np.isfinite(query).all()
        return x, query


def make_component(view, full):
    audio, classes, y, checked, val = load_training()
    fit = np.arange(len(audio)) if full else np.setdiff1d(np.arange(len(audio)), val)
    test = read_archive("FSDKaggle2018.audio_test") if full else None
    if view in ("combined", "fusion"):
        train_parts, query_parts = [], []
        mixture = [("texture", .50), ("patches", .40), ("fisher", .10)]
        if view == "fusion":
            mixture = [("texture", .30), ("covariance", .30), ("patches", .25), ("tonal", .15)]
        for name, weight in mixture:
            x, query = split_features(name, fit, val, audio, test)
            if name in ("texture", "covariance", "tonal"):
                scaler = StandardScaler().fit(x)
                x = scaler.transform(x).clip(-5., 5.)
                query = scaler.transform(query).clip(-5., 5.)
            train_parts.append(np.sqrt(weight).astype("float32") * normalize(x))
            query_parts.append(np.sqrt(weight).astype("float32") * normalize(query))
        x = np.hstack(train_parts)
        query = np.hstack(query_parts)
        del train_parts, query_parts
    else:
        x, query = split_features(view, fit, val, audio, test)
    predictions = predict_view(x, query, y[fit], checked[fit], len(classes), view in ("texture", "tonal", "covariance"), view != "fisher")
    phase = "full" if full else "validation"
    kw = {"names":test.names.to_numpy(dtype=str)} if full else {"val_idx":val}
    np.savez(OUT / f"{view}_{phase}.npz", classes=classes, **predictions, **kw)
    if not full:
        print(view, {name: map3(probs, y[val]) for name, probs in predictions.items()}, flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=["validation","refit"])
    ap.add_argument("views", nargs="+", choices=VIEWS)
    args=ap.parse_args()
    OUT.mkdir(exist_ok=True, parents=True)
    for view in args.views:
        make_component(view, args.stage=="refit")


if __name__=="__main__":
    main()
