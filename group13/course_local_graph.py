"""Build course-only acoustic neighbourhoods, including bidirectional patch matching."""
from course_input import course_data
from pathlib import Path
import os
for key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[key] = "4"

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import eigsh
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize, StandardScaler

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results/course_local_graph"
FEATURES = ROOT / "results/native_upgrade/course/features"


def neighbours(x, count=70):
    values, indices = [], []
    for start in range(0, len(x), 160):
        block = 1 - x[start:start+160] @ x.T
        row = np.arange(len(block))
        block[row, row+start] = np.inf
        idx = np.argpartition(block, count-1, axis=1)[:, :count]
        values.append(np.take_along_axis(block, idx, axis=1).clip(1e-5))
        indices.append(idx)
    return np.vstack(indices), np.vstack(values)


def affinity(idx, distance, k):
    near = np.argpartition(distance, k-1, axis=1)[:, :k]
    d = np.take_along_axis(distance, near, axis=1)
    idx = np.take_along_axis(idx, near, axis=1)
    # Scale an edge relative to the neighbourhoods of both endpoints.
    width = np.maximum(np.median(d, axis=1), 1e-5)
    w = np.exp(-2 * d / np.sqrt(width[:,None] * width[idx]))
    a = sparse.csr_matrix((w.ravel(), idx.ravel(), np.arange(0, (len(d)+1)*k,k)),
                          shape=(len(d),len(d)))
    return (a + a.T + 2*a.minimum(a.T)).tocsr()


def clip_view(name, unchecked):
    if name.startswith("fisher"):
        with np.load(ROOT/"cache"/f"fisher_audio-v1_{name[6:]}.npz") as data:
            return normalize(data["x"], copy=False)
    if name == "summary":
        x = np.load(ROOT/"cache/audio-v1/matrix.npy")
    else:
        x = np.concatenate([np.load(FEATURES/f"{s}_{name}.npy") for s in ("train","test")])
    if name != "dictionary":
        x = StandardScaler().fit(x[unchecked]).transform(x).clip(-6,6)
    return normalize(x.astype(np.float32), copy=False)


def patch_distance(unchecked, shortlists):
    # Each patch contains local mel-band levels and temporal changes. A brief
    # event can match even when it occurs at different times in two recordings.
    rng = np.random.default_rng(729)
    events = [np.load(FEATURES/f"{s}_events.npy", mmap_mode="r") for s in ("train","test")]
    rows = rng.choice(unchecked, 46000)
    frames = events[0][rows, rng.integers(96, size=len(rows))].astype("float32")
    scale = StandardScaler().fit(frames)
    pca = PCA(n_components=48, random_state=729, svd_solver="randomized").fit(scale.transform(frames))
    patches = []
    for source in events:
        for i in range(0,len(source),128):
            chunk = source[i:i+128, 1::3].astype("float32")
            projected = pca.transform(scale.transform(chunk.reshape(-1,chunk.shape[-1])))
            projected /= pca.explained_variance_**.25
            patches.append(normalize(projected).reshape(len(chunk),32,-1))
    x = np.vstack(patches)
    idx_all, dist_all = [], []
    for i in range(len(x)):
        idx = np.unique(np.concatenate([view[i] for view in shortlists]))
        sim = x[idx] @ x[i].T
        a, b = sim.max(1), sim.max(2)
        # Pool both directions to penalize incidental matches to a single frame.
        s = .35*(a.mean(1)+b.mean(1)) + .15*(np.sort(a,axis=1)[:,16:].mean(1)
                                                + np.sort(b,axis=1)[:,16:].mean(1))
        keep = np.argsort(-s)[:40]
        idx_all.append(idx[keep])
        dist_all.append(1-s[keep])
    return np.stack(idx_all), np.stack(dist_all)


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(course_data(ROOT) / 'train.csv',dtype={"id":str})
    test = pd.read_csv(course_data(ROOT) / 'test.csv',dtype=str)
    for split, ids in [("train",train.id),("test",test.id)]:
        if not np.array_equal(ids,np.load(FEATURES/f"{split}_names.npy")):
            raise ValueError("Course feature ID order differs")
    unchecked = np.flatnonzero(train.checked.eq(0))
    graphs, shortlists = [], []
    for name in ["summary","fisher8","fisher32","fisher64","dictionary","texture","covariance"]:
        cache = OUT/f"near_{name}.npz"
        if not cache.exists():
            idx, d = neighbours(clip_view(name,unchecked))
            np.savez(cache, idx=idx, d=d)
        with np.load(cache) as z:
            idx, d = z["idx"], z["d"]
        order = np.argsort(d,axis=1)[:,:45]
        shortlists.append(np.take_along_axis(idx,order,axis=1))
        graph = affinity(idx,d,18)
        graphs.append(graph/graph.sum()*len(idx))
        print("graph", name, flush=True)
    cache = OUT/"near_segments.npz"
    if not cache.exists():
        idx,d = patch_distance(unchecked,shortlists)
        np.savez(cache,idx=idx,d=d)
    with np.load(cache) as z:
        patch = affinity(z["idx"],z["d"],14)
    # Also connect clips which repeatedly share nearest neighbours across views.
    sim = sum(graphs)/len(graphs)
    tri = sim @ normalize(sim, norm="l1")
    shared = tri.multiply(sim.sign()).tocsr()
    shared = (shared+shared.T)*.5
    graph = (.52*sim + .30*patch/patch.sum()*len(idx)
             + .18*shared/shared.sum()*len(idx)).tocsr()
    sparse.save_npz(OUT/"graph.npz",graph)
    deg = sparse.diags(1/np.sqrt(np.maximum(graph.sum(1).A1,1e-8)))
    eig, vectors = eigsh(deg @ graph @ deg, k=150, which="LA",
                         v0=np.random.default_rng(10).normal(size=len(idx)).astype(np.float32),tol=1e-5)
    np.savez(OUT/"diffusion.npz", eig=eig, vec=vectors)


if __name__ == "__main__":
    build()
