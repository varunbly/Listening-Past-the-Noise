"""Compute fixed time-frequency, modulation and periodicity features from the release WAVs."""
import argparse
import io
import os
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path
import wave
from zipfile import ZipFile

import numpy as np
from scipy.fft import dct, irfft, rfft
from scipy.ndimage import uniform_filter1d
from scipy.sparse import csr_matrix


BASE = Path(__file__).resolve().parent
RATE = 44100
FRAME = 2048
HOP = 882
BANDS = 72


@lru_cache(maxsize=4)
def filters(frame=FRAME, bands=BANDS, rate=RATE):
    frequencies = np.fft.rfftfreq(frame, 1 / rate)
    limits = np.log1p(np.array([30, min(20000, rate * .475)]) / 700)
    edges = 700 * np.expm1(np.linspace(limits[0], limits[1], bands + 2))
    rise = (frequencies - edges[:-2, None]) / np.diff(edges)[:-1, None]
    fall = (edges[2:, None] - frequencies) / np.diff(edges)[1:, None]
    mel = np.maximum(np.minimum(rise, fall), 0)
    mel /= np.maximum(mel.sum(axis=1, keepdims=True), 1e-12)
    window = np.hanning(frame).astype('float32')
    return csr_matrix(mel.astype('float32')), window


def summaries(values):
    return [values.mean(0), values.std(0), *np.percentile(values, [10, 50, 90], axis=0)]


def modulation(envelopes):
    """Windowed modulation spectra capture sustained pulses and repeated attacks."""
    width = 128
    if len(envelopes) < width:
        envelopes = np.pad(envelopes, ((0, width - len(envelopes)), (0, 0)), mode='edge')
    windows = np.lib.stride_tricks.sliding_window_view(envelopes, width, axis=0)[::width // 2]
    centered = (windows - windows.mean(axis=-1, keepdims=True)) * np.hanning(width).astype('float32')
    power = np.abs(rfft(centered, axis=-1)) ** 2
    edges = [1, 3, 6, 11, 19, 32, 49, 65]
    return [np.log1p(power[:, :, a:b].mean(axis=(0, 2))) for a, b in zip(edges, edges[1:])]


def periodicity(power, energy, rate=RATE):
    """Pool the strengths and pitch distribution of prominent autocorrelation peaks."""
    # Compression whitens the spectrum enough to expose overtones.
    correlation = irfft(power[::3] ** .65, axis=1)
    correlation /= np.maximum(correlation[:, :1], 1e-12)
    ranges = np.rint(np.array([28, 55, 110, 220, 440, 736]) * (rate / RATE)).astype(int)
    lags = np.arange(ranges[0], ranges[-1])
    middle = correlation[:, lags]
    local_peaks = (middle > correlation[:, lags - 1]) & (middle >= correlation[:, lags + 1])
    best = (middle * local_peaks).argmax(axis=1)
    clarity = np.maximum(middle[np.arange(len(middle)), best], 0)
    # Downweight near-silent frames. Weight stable periodic frames more strongly.
    loudness = energy[::3] / max(float(energy.max()), 1e-10)
    weights = clarity ** 2 * np.sqrt(loudness)
    midi = 69 + 12 * np.log2((rate / lags[best]) / 440)
    pitches = np.histogram(midi, bins=np.arange(33, 96, 2), weights=weights)[0]
    pitches /= max(weights.sum(), 1e-8)
    chroma = np.bincount(np.rint(midi).astype(int) % 12, weights=weights, minlength=12)
    chroma /= max(weights.sum(), 1e-8)
    octave_strength = np.stack([correlation[:, a:b].max(1) for a, b in zip(ranges, ranges[1:])], axis=1)
    return [pitches, chroma, *summaries(octave_strength), np.percentile(clarity, [10, 50, 90])]


def waveform_features(samples, rate=RATE):
    width, hop = round(FRAME * rate / RATE), round(HOP * rate / RATE)
    mel, window = filters(width, BANDS, rate)
    if len(samples) < width:
        samples = np.pad(samples, (0, width - len(samples)))
    frames = np.lib.stride_tricks.sliding_window_view(samples, width)[::hop]
    spectrum = rfft(frames * window, axis=1)
    power = spectrum.real ** 2 + spectrum.imag ** 2
    energy = power.sum(1)
    mel_energy = (mel @ power.T).T
    peak = max(float(mel_energy.max()), 1e-10)
    logmel = np.maximum(10 * np.log10(np.maximum(mel_energy / peak, 1e-9)), -80)
    # Each 20-ms frame has a log-mel spectral envelope and a cepstral description.
    features = [*summaries(logmel)]
    cepstra = dct(logmel, type=2, axis=1, norm='ortho')[:, :24]
    features.extend(summaries(cepstra))
    centered = cepstra - cepstra.mean(0)
    covariance = centered.T @ centered / len(centered)
    upper = covariance[np.triu_indices(covariance.shape[0], k=1)]
    features.append(np.sign(upper) * np.sqrt(np.abs(upper)))

    pooled = logmel.reshape(len(logmel), BANDS // 3, 3).mean(2)
    features.extend(modulation(pooled))
    # Haar differences across time, frequency and diagonals describe local contours.
    for step, bands in [(1, 1), (4, 2), (12, 3)]:
        smooth = uniform_filter1d(pooled, step, axis=0, mode='nearest')
        smooth = uniform_filter1d(smooth, bands, axis=1, mode='nearest')
        if len(smooth) <= step:
            smooth = np.pad(smooth, ((0, step + 1 - len(smooth)), (0, 0)), mode='edge')
        later, earlier = smooth[step:], smooth[:-step]
        for delta in [later - earlier, later[:, bands:] - later[:, :-bands],
                      later[:, bands:] - earlier[:, :-bands], later[:, :-bands] - earlier[:, bands:]]:
            features.extend([np.maximum(delta, 0).mean(0), np.maximum(-delta, 0).mean(0)])
    # Spectral flatness and prominence by octave separate tonal and noise-like sounds.
    octave_edges = [edge for edge in (1, 5, 10, 20, 40, 80, 160, 320, 640) if edge < power.shape[1]] + [power.shape[1]]
    flatness, prominence = [], []
    for low, high in zip(octave_edges, octave_edges[1:]):
        subband = np.maximum(power[:, low:high], 1e-9)
        arithmetic = np.log(subband.mean(1))
        flatness.append(np.log(subband).mean(1) - arithmetic)
        prominence.append(np.log(subband.max(1)) - arithmetic)
    features.extend(summaries(np.array(flatness + prominence).T))
    features.extend(periodicity(power, energy, rate))
    active = energy > max(float(energy.max()) * .01, 1e-10)
    positions = np.arange(len(energy)) / max(len(energy) - 1, 1)
    features.extend([np.array([np.log1p(len(samples) / rate), active.mean(),
                              positions[energy.argmax()], np.dot(positions, energy) / max(energy.sum(), 1e-9)])])
    result = np.concatenate(features).astype('float32')
    if not np.isfinite(result).all():
        raise ValueError('Non-finite texture descriptor')
    return result


@lru_cache(maxsize=2)
def open_archive(path):
    return ZipFile(path)


def batch_features(args):
    path, members, descriptor = args
    archive = open_archive(path)
    result = []
    for name in members:
        with wave.open(io.BytesIO(archive.read(name))) as wav:
            if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) != (1, 2, RATE):
                raise ValueError(f'Unexpected WAV format in {name}')
            audio = np.frombuffer(wav.readframes(wav.getnframes()), dtype='<i2').astype('float32') / 32768
        result.append(descriptor(audio))
    return np.stack(result)


def extract_archives(descriptor, output_name, description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--archives', type=Path, default=BASE.parent)
    parser.add_argument('--output', type=Path, default=BASE / 'results/freesound_features' / output_name)
    args = parser.parse_args()
    # Keep each worker on one BLAS thread during the small cepstral covariance product.
    for key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
        os.environ[key] = '1'
    output = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for split in ('train', 'test'):
            path = args.archives / f'FSDKaggle2018.audio_{split}.zip'
            with ZipFile(path) as archive:
                names = [name for name in archive.namelist() if name.lower().endswith('.wav')]
            if len(set(names)) != len(names) or not names:
                raise ValueError(f'Invalid WAV manifest in {path}')
            batches = ((path, names[i:i + 16], descriptor) for i in range(0, len(names), 16))
            blocks = []
            for i, block in enumerate(pool.map(batch_features, batches), start=1):
                blocks.append(block)
                if i % 32 == 0 or i * 16 >= len(names):
                    print(f'{split} {min(i * 16, len(names))}/{len(names)}', flush=True)
            output[split] = np.concatenate(blocks)
            output[split + '_names'] = np.asarray(names)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix('.tmp.npz')
    np.savez(temporary, **output)
    temporary.replace(args.output)
    print(f'{args.output} {output["train"].shape} {output["test"].shape}', flush=True)


if __name__ == '__main__':
    extract_archives(waveform_features, 'texture.npz', __doc__)
