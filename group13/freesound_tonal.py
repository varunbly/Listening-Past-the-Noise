"""Extract pitch and pitch-aligned timbre descriptors from the full native WAV archives."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from time import perf_counter
import wave
from zipfile import ZipFile

import numpy as np
from scipy.fft import rfft
from scipy.ndimage import gaussian_filter1d
from scipy.sparse import csr_matrix

from train_freesound_local import BASE, read_archive

RATE = 44100
FFT = 16384
BINS_PER_OCTAVE = 36
FIRST_MIDI = 26
OUT = BASE / "results" / "freesound_features" / "tonal.npz"


@lru_cache(maxsize=1)
def filterbank(rate=RATE):
    """Each log band spans two thirds of a semitone, centered a third of a semitone apart."""
    bottom = 440 * 2 ** ((FIRST_MIDI - 69) / 12)
    width = round(FFT * rate / RATE)
    count = int(np.floor(BINS_PER_OCTAVE * np.log2(min(19000, .47 * rate) / bottom)))
    centers = bottom * 2 ** (np.arange(count) / BINS_PER_OCTAVE)
    fft_hz = np.fft.rfftfreq(width, 1 / rate)
    indices, values, offsets = [], [], [0]
    for center in centers:
        low, high = center * 2 ** (-1 / BINS_PER_OCTAVE), center * 2 ** (1 / BINS_PER_OCTAVE)
        idx = np.where((fft_hz > low) & (fft_hz < high))[0]
        if len(idx) < 2:
            fbin = center * width / rate
            idx = np.array([int(fbin), int(fbin) + 1])
            weights = np.array([1 - fbin % 1, fbin % 1])
        else:
            weights = np.minimum((fft_hz[idx] - low) / (center - low),
                                 (high - fft_hz[idx]) / (high - center))
        indices.extend(idx)
        values.extend(weights / weights.sum())
        offsets.append(len(indices))
    bank = csr_matrix((np.array(values, dtype=np.float32), indices, offsets),
                      shape=(count, width // 2 + 1))
    window = np.hanning(width).astype(np.float32)
    return bank, window


@lru_cache(maxsize=1)
def harmonics():
    # Up to eight harmonics of fundamentals from MIDI 32 to 96, every third semitone.
    pitch = np.arange((32 - FIRST_MIDI) * 3, (96 - FIRST_MIDI) * 3 + 1)
    harmonics = np.arange(1, 9)
    idx = pitch[:, None] + np.rint(BINS_PER_OCTAVE * np.log2(harmonics)).astype(int)
    between = pitch[:, None] + np.rint(BINS_PER_OCTAVE * np.log2(harmonics + .5)).astype(int)
    weight = harmonics ** -.78
    return pitch, idx, between, (weight / weight.sum()).astype(np.float32)


def tonal_features(samples, rate=RATE):
    bank, window = filterbank(rate)
    # A fixed cap per clip bounds cost while spanning the entire recording.
    count = max(1, min(48, int(np.ceil(len(samples) / (2700 * rate / RATE)))))
    indices = np.linspace(0, len(samples) - 1, count).astype(int)
    padded = np.pad(samples, (len(window) // 2, len(window) // 2))
    frames = np.lib.stride_tricks.sliding_window_view(padded, len(window))[indices]
    spectrum = rfft(frames * window)
    power = np.square(spectrum.real) + np.square(spectrum.imag)
    level = power.sum(1)
    active = level >= max(float(level.max()) * .003, 1e-12)
    if not active.any():
        active[:] = True
    mel = (bank @ power[active].T).T
    floor = max(float(mel.max()) * 1e-9, 1.e-12)
    log_band = 10 * np.log10(np.maximum(mel, floor))
    log_band -= np.max(log_band, axis=1, keepdims=True)
    envelope = gaussian_filter1d(log_band, 1.6, axis=1, mode="nearest")
    features = [envelope[:, 1::3].mean(0), envelope[:, 1::3].std(0),
                np.percentile(envelope[:, 1::3], 85, axis=0)]
    # Whiten the broad spectral envelope before comparing harmonic combs.
    white = log_band - gaussian_filter1d(log_band, 6, axis=1, mode="nearest")
    pitch, on, off, weight = harmonics()
    # For lower-rate material, renormalize harmonic templates below Nyquist.
    if off.max() >= log_band.shape[1]:
        weights = weight * (off < log_band.shape[1])
        weights /= np.maximum(weights.sum(1, keepdims=True), 1e-8)
        on = on.clip(max=log_band.shape[1]-1)
        off = off.clip(max=log_band.shape[1]-1)
        salience = np.sum((.70 * white[:, on] + .30 * (log_band[:, on] - log_band[:, off])) * weights, axis=2)
    else:
        salience = ((.70 * white[:, on] + .30 * (log_band[:, on] - log_band[:, off])) @ weight)
    # An inaudible fundamental is plausible, but penalize combs with negligible low harmonics.
    audibility = log_band[:, on[:, :3]].max(axis=2)
    salience += .24 * np.minimum(audibility + 24, 0)
    pitch_scores = gaussian_filter1d(salience, 1.1, axis=1)[:, ::3]
    features += [pitch_scores.mean(0), pitch_scores.std(0), np.percentile(pitch_scores, 90, axis=0)]
    best = np.argmax(salience, axis=1)
    row = np.arange(len(best))
    confidence = np.maximum(salience[row, best], .01)
    mass = confidence / confidence.sum()
    histogram = np.bincount(np.rint(best / 3).astype(int), weights=mass, minlength=pitch_scores.shape[1])
    # Smooth the pitch histogram to avoid hard boundaries between neighbouring notes.
    features.append(gaussian_filter1d(histogram, 1.5))
    # Aligned overtones distinguish instruments that share a fundamental frequency.
    aligned = log_band[row[:, None], on[best]]
    aligned -= aligned.max(1, keepdims=True)
    contrast = white[row[:, None], on[best]]
    odd_even = aligned[:, 0::2] - aligned[:, 1::2]
    aligned_features = np.column_stack((aligned, contrast, odd_even))
    features += [np.average(aligned_features, axis=0, weights=mass), aligned_features.std(0),
                 *np.percentile(aligned_features, [15, 85], axis=0)]
    features += [aligned_features[:max(1, len(best)//3)].mean(0),
                 aligned_features[-max(1, len(best)//3):].mean(0)]
    prominence = np.maximum(white, 0)
    octave_width = BINS_PER_OCTAVE
    for lo in range(0, prominence.shape[1], octave_width):
        a = prominence[:, lo:lo + octave_width]
        features.append(np.array([a.mean(), a.std(), a.max(1).mean(), np.mean(a > 4)]))
    movement = np.abs(np.diff(best, prepend=best[0])) / 3
    features.append(np.array([np.average(best / 3 + 32, weights=mass), np.std(best / 3),
                              movement.mean(), np.mean(movement > 2),
                              np.mean(confidence), np.std(confidence), np.log1p(len(samples) / rate)]))
    return np.concatenate(features).astype(np.float32)


def decode_and_extract(data):
    with wave.open(BytesIO(data)) as recording:
        if recording.getframerate() != RATE or recording.getsampwidth() != 2:
            raise ValueError("Expected 44.1 kHz, 16-bit PCM audio")
        samples = np.frombuffer(recording.readframes(recording.getnframes()), dtype="<i2").astype("float32")
        if recording.getnchannels() > 1:
            samples = samples.reshape(-1, recording.getnchannels()).mean(1)
    if not len(samples):
        raise ValueError("Empty WAV")
    return tonal_features(samples / 32768)


def extract(stem, workers):
    names = read_archive(stem).names.to_numpy(dtype=str)
    result = []
    started = perf_counter()
    with ZipFile(BASE.parent / (stem + ".zip")) as archive, ThreadPoolExecutor(workers) as pool:
        for begin in range(0, len(names), 32):
            payload = [archive.read(name) for name in names[begin:begin + 32]]
            result.extend(pool.map(decode_and_extract, payload))
            if (begin and begin % 1024 == 0) or len(result) == len(names):
                print(f"{stem} {len(result)}/{len(names)} ({perf_counter() - started:.1f}s)", flush=True)
    matrix = np.stack(result)
    if not np.isfinite(matrix).all():
        raise ValueError(f"Invalid tonal descriptors in {stem}")
    return matrix, names


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    train, train_names = extract("FSDKaggle2018.audio_train", args.workers)
    test, test_names = extract("FSDKaggle2018.audio_test", args.workers)
    OUT.parent.mkdir(exist_ok=True, parents=True)
    temporary = OUT.with_suffix(".writing.npz")
    np.savez(temporary, train=train, train_names=train_names, test=test, test_names=test_names)
    temporary.replace(OUT)
    print(OUT, train.shape, test.shape, flush=True)


if __name__ == "__main__":
    main()
