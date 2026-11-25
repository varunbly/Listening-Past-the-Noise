"""Extract log-covariance and harmonic/percussive envelopes at two time resolutions."""
import numpy as np
from scipy.fft import dct, rfft
from scipy.ndimage import correlate1d, median_filter

from freesound_texture import extract_archives, filters


def log_covariance(rows):
    """Represent shrinkage covariance in log-Euclidean tangent coordinates."""
    centered = rows - rows.mean(0)
    covariance = centered.T @ centered / max(len(rows) - 1, 1)
    shrinkage = .08
    diagonal = max(covariance.trace() / len(covariance), 1.)
    covariance *= 1 - shrinkage
    covariance.flat[::len(covariance) + 1] += shrinkage * diagonal
    values, vectors = np.linalg.eigh(covariance)
    logged = (vectors * np.log(np.maximum(values, 1e-6))) @ vectors.T
    i, j = np.triu_indices(len(covariance))
    return logged[i, j] * np.where(i == j, 1., np.sqrt(2.))


def envelope_summary(energy, active):
    db = np.clip(10 * np.log10(np.maximum(energy, 1e-9)), -80., 0.)
    frames = db[active]
    # The proportion of each sound type and its evolving spectral shape provide
    # distinct cues for sustained notes, speech, machinery and short impacts.
    return [frames.mean(0), frames.std(0), *np.percentile(frames, [20, 80], axis=0),
            np.abs(np.diff(db, axis=0, prepend=db[:1])).mean(0)]


def resolution_features(audio, width, hop, bands, rate=44100):
    mel, window = filters(width, bands, rate)
    if len(audio) < width:
        audio = np.pad(audio, (0, width - len(audio)))
    frames = np.lib.stride_tricks.sliding_window_view(audio, width)[::hop]
    fft = rfft(frames * window, axis=1)
    power = fft.real ** 2 + fft.imag ** 2
    energy = power.sum(1)
    active = energy >= max(float(energy.max()) * .0003, 1e-10)
    if not active.any():
        active[:] = True
    mel_energy = (mel @ power.T).T
    mel_energy /= max(float(mel_energy.max()), 1e-10)
    db = np.maximum(10 * np.log10(np.maximum(mel_energy, 1e-9)), -80.)
    cepstra = np.ascontiguousarray(dct(db, type=2, axis=1, norm='ortho')[:, :20])
    # The zeroth coefficient sums all band energies. Scale it so shrinkage treats
    # the other cepstral dimensions and their derivatives more evenly.
    cepstra[:, 0] *= .25
    delta = correlate1d(cepstra, np.array([-2, -1, 0, 1, 2]) / 10., axis=0, mode='nearest')
    joint = np.column_stack((cepstra, 3. * delta))
    result = [log_covariance(joint[active]), joint[active].mean(0)]
    active_ids = np.flatnonzero(active)
    # Difference between onset and release distinguishes similar timbres.
    count = max(1, len(active_ids) // 4)
    result.append(cepstra[active_ids[:count]].mean(0) - cepstra[active_ids[-count:]].mean(0))
    # Soft median masks compare persistence across time with breadth across frequency.
    harmonic = median_filter(mel_energy, size=(11, 1), mode='nearest')
    percussive = median_filter(mel_energy, size=(1, 9), mode='nearest')
    harmonic **= 2
    percussive **= 2
    mask = harmonic / np.maximum(harmonic + percussive, 1e-12)
    result.extend(envelope_summary(mel_energy * mask, active))
    result.extend(envelope_summary(mel_energy * (1 - mask), active))
    result.extend([mask[active].mean(0), mask[active].std(0)])
    return result


def waveform_covariance(audio, rate=44100):
    # Short frames preserve impacts. Long frames resolve narrow, low-frequency harmonics.
    # Both describe continuous time on a fixed hop throughout each original recording.
    hop = round(882 * rate / 44100)
    result = resolution_features(audio, round(1024 * rate / 44100), hop, 48, rate)
    result.extend(resolution_features(audio, round(4096 * rate / 44100), hop, 72, rate))
    features = np.concatenate(result).astype('float32')
    if not np.isfinite(features).all():
        raise ValueError('Non-finite covariance descriptor')
    return features


if __name__ == '__main__':
    extract_archives(waveform_covariance, 'covariance.npz', __doc__)
