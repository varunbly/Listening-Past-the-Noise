"""Build rate-aware acoustic features using only the selected dataset's recordings."""
from course_input import course_data
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from io import BytesIO
import os
from pathlib import Path
import wave
from zipfile import ZipFile

for variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
import pandas as pd
from scipy.fft import rfft
from scipy.ndimage import uniform_filter1d
from scipy.special import softmax
from sklearn.preprocessing import normalize

from freesound_covariance import waveform_covariance
from freesound_texture import filters, waveform_features
from freesound_tonal import tonal_features
from freesound_multiview import load_training
from train_freesound_local import BASE, read_archive

ROOT = BASE / "results/native_upgrade"
COURSE = course_data(BASE)
VIEWS = ["texture", "covariance", "tonal"]
PATCH_COUNT = 96


def manifest(dataset, split):
    if dataset == "course":
        frame = pd.read_csv(COURSE / f"{split}.csv", dtype={"id": str})
        return frame.id.to_numpy(str)
    return read_archive("FSDKaggle2018.audio_" + split).names.to_numpy(str)


@lru_cache(maxsize=2)
def archive(split):
    return ZipFile(BASE.parent / f"FSDKaggle2018.audio_{split}.zip")


def load_wav(dataset, split, name):
    if dataset == "course":
        source = str(COURSE / "audio/audio" / split / (name + ".wav"))
    else:
        source = BytesIO(archive(split).read(name))
    with wave.open(source) as f:
        if f.getsampwidth() != 2 or f.getnchannels() != 1:
            raise ValueError(f"Expected mono PCM16 at {name}")
        rate = f.getframerate()
        samples = np.frombuffer(f.readframes(f.getnframes()), dtype="<i2").astype(np.float32) / 32768
    if not len(samples) or rate != (22050 if dataset == "course" else 44100):
        raise ValueError(f"Unexpected audio length or rate at {name}")
    return samples, rate


def event_frames(samples, rate):
    """Collect local timbre and two timescales of change across active portions of each file."""
    width = round(2048 * rate / 44100)
    # The filter spacing, frequency ceiling and hop follow the recording's sample rate.
    mel, window = filters(width, 48, rate)
    signal = np.pad(samples, width // 2)
    frames = np.lib.stride_tricks.sliding_window_view(signal, width)[::round(.0125*rate)]
    power = np.abs(rfft(frames * window, axis=1))**2
    bands = (mel @ power.T).T
    db = np.clip(10*np.log10(np.maximum(bands / max(float(bands.max()), 1e-12), 1e-9)), -80, 0)
    envelope = db.max(1)
    active = np.flatnonzero(envelope > -43)
    if not len(active):
        active = np.arange(len(envelope))
    # Equal-duration sampling within active frames preserves brief events in long, quiet clips.
    ids = active[np.rint(np.linspace(0, len(active)-1, PATCH_COUNT)).astype(int)]
    short = uniform_filter1d(db, 3, axis=0, mode="nearest")
    long = uniform_filter1d(db, 9, axis=0, mode="nearest")
    prev, nxt = np.maximum(ids-3, 0), np.minimum(ids+3, len(db)-1)
    before, after = np.maximum(ids-11, 0), np.minimum(ids+11, len(db)-1)
    center = short[ids]
    return np.hstack((center-center.mean(1, keepdims=True), .65*(short[nxt]-short[prev]),
                      .5*(long[after]-long[before]), .5*(center - .5*(long[before]+long[after])))).astype("float16")


def describe_batch(args):
    dataset, split, names = args
    output = {"events": []}
    if dataset == "course":
        output.update({view: [] for view in VIEWS})
    for name in names:
        samples, rate = load_wav(dataset, split, name)
        output["events"].append(event_frames(samples, rate))
        if dataset == "course":
            output["texture"].append(waveform_features(samples, rate))
            output["covariance"].append(waveform_covariance(samples, rate))
            output["tonal"].append(tonal_features(samples, rate))
    return {key: np.stack(value) for key, value in output.items()}


def extract(dataset, workers):
    directory = ROOT / dataset / "features"
    directory.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(workers) as pool:
        for split in ("train", "test"):
            names = manifest(dataset, split)
            keys = ["events", *VIEWS] if dataset == "course" else ["events"]
            final = [directory / f"{split}_{key}.npy" for key in keys]
            if all(path.is_file() for path in final):
                continue
            batch_size = 24
            batches = [(dataset, split, names[i:i+batch_size]) for i in range(0, len(names), batch_size)]
            mappings = {}
            count = 0
            for i, batch in enumerate(pool.map(describe_batch, batches)):
                size = len(batch["events"])
                for key, values in batch.items():
                    if i == 0:
                        mappings[key] = np.lib.format.open_memmap(
                            directory / f"{split}_{key}.pending", "w+", values.dtype, (len(names), *values.shape[1:]))
                    if not np.isfinite(values).all():
                        raise ValueError(f"Nonfinite {key} in {split} at batch {i}")
                    mappings[key][count:count+size] = values
                count += size
                if i % 20 == 0 or count == len(names):
                    print(f"{dataset} {split}: extracted {count}/{len(names)}", flush=True)
            if count != len(names):
                raise RuntimeError("Incomplete extraction")
            for key in keys:
                mappings[key].flush()
                del mappings[key]
                (directory / f"{split}_{key}.pending").replace(directory / f"{split}_{key}.npy")
            np.save(directory / f"{split}_names.npy", names)


def learn_centers(frames, unchecked):
    # Only unchecked training clips supply dictionary samples. They are present in
    # every verification fold and are disjoint from all held-out and test audio.
    rng = np.random.default_rng(2146)
    indices = rng.choice(unchecked, 90000)
    samples = frames[indices, rng.integers(PATCH_COUNT, size=len(indices))].astype(np.float32)
    mean = samples.mean(0)
    samples -= mean
    spectrum, vectors = np.linalg.eigh(samples.T @ samples / len(samples))
    top = np.argsort(spectrum)[-48:][::-1]
    whitening = vectors[:, top] / np.sqrt(spectrum[top] + 2.)
    x = normalize(samples @ whitening)
    centers = x[rng.choice(len(x), 160, replace=False)].copy()
    for _ in range(24):
        sums = np.zeros_like(centers)
        for begin in range(0, len(x), 2048):
            block = x[begin:begin+2048]
            membership = (block @ centers.T).argmax(1)
            # bincount performs the reduction with each row contributing once.
            for j in range(x.shape[1]):
                sums[:, j] += np.bincount(membership, weights=block[:,j], minlength=len(centers))
        small = np.linalg.norm(sums, axis=1) < 1e-9
        sums[small] = x[rng.choice(len(x), small.sum(), replace=False)]
        centers = normalize(sums)
    return mean, whitening, centers


def encode_events(frames, dictionary):
    mean, whitening, centers = dictionary
    collected = []
    for begin in range(0, len(frames), 32):
        b = frames[begin:begin+32].astype("float32")
        projected = (b-mean) @ whitening
        shape = projected.shape
        x = normalize(projected.reshape(-1, shape[-1])).reshape(shape)
        response = x @ centers.T
        memberships = softmax(response*18., axis=2)
        weights = memberships.transpose(0,2,1) / PATCH_COUNT
        mass = weights.sum(2)
        first = weights @ x - mass[:,:,None]*centers
        # Relative frequency, strongest responses and early/late changes complement
        # the first-order residual distribution.
        half = PATCH_COUNT // 2
        direction = memberships[:,half:].mean(1)**.5 - memberships[:,:half].mean(1)**.5
        components = [normalize((np.sign(first)*np.sqrt(np.abs(first))).reshape(len(b),-1)),
                      .60*normalize(np.sqrt(mass)), .42*normalize(response.max(1).clip(0)**3),
                      .22*normalize(direction)]
        collected.append(np.column_stack(components))
    return np.vstack(collected)


def encode(dataset):
    directory = ROOT / dataset / "features"
    if dataset == "course":
        unchecked = np.flatnonzero(pd.read_csv(COURSE / "train.csv").checked.eq(0))
    else:
        unchecked = np.flatnonzero(~load_training()[3])
    def events(split):
        if not np.array_equal(np.load(directory / f"{split}_names.npy"), manifest(dataset, split)):
            raise ValueError("Manifest differs from the extracted recording order")
        return np.load(directory / f"{split}_events.npy", mmap_mode="r")
    dictionary = learn_centers(events("train"), unchecked)
    np.savez(directory / "dictionary.npz", mean=dictionary[0], whitening=dictionary[1], centers=dictionary[2])
    for split in ("train", "test"):
        matrix = encode_events(events(split), dictionary)
        if not np.isfinite(matrix).all():
            raise ValueError("Nonfinite event encoding")
        temporary = directory / f"{split}_dictionary.pending.npy"
        np.save(temporary, matrix)
        temporary.replace(directory / f"{split}_dictionary.npy")
        print(dataset, split, "encoded", matrix.shape, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=["course", "freesound"])
    parser.add_argument("--workers", default=3, type=int)
    args = parser.parse_args()
    extract(args.dataset, args.workers)
    encode(args.dataset)
