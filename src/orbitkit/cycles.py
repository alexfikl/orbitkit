# SPDX-FileCopyrightText: 2025 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np

from orbitkit.typing import Array1D, Array2D
from orbitkit.utils import module_logger

log = module_logger(__name__)


# {{{ power spectrum density


@dataclass(frozen=True, slots=True)
class PowerSpectrumDensity:
    harmonic_energy: float
    """Energy at all the harmonics of the mean power spectrum density."""
    total_energy: float
    """Total energy in the signal."""

    freq: Array1D[np.floating[Any]]
    """Frequencies of the last computed power spectrum density."""
    psd: Array1D[np.floating[Any]]
    """The last computed power spectrum density."""

    def is_periodic(self, eps: float = 1.0e-3) -> bool:
        """Check if the corresponding time series is periodic based on the PSD."""
        total_energy = 1.0 if self.total_energy < 1.0e-8 else self.total_energy
        delta = 1 - self.harmonic_energy / total_energy

        log.info("Periodic: delta %.8e (eps %.8e)", delta, eps)
        return bool(delta < eps)


def make_windows(
    n: int,
    nwindows: int,
    length: int,
    overlap: float = 0.5,
) -> Iterator[tuple[int, int]]:
    step = int((1 - overlap) * length)
    w_ends = [n - i * step for i in reversed(range(nwindows))]
    w_starts = [max(0, e - length) for e in w_ends]

    return zip(w_starts, w_ends, strict=True)


def make_harmonic_mask(
    f: Array1D[np.number[Any]],
    f0: float,
    *,
    nharmonics: int = 5,
    binwidth: int = 1,
) -> Array1D[np.bool]:
    df = f[1] - f[0]
    mask = np.zeros(f.shape, dtype=np.bool)
    for k in range(1, nharmonics + 1):
        mask |= np.abs(f - k * f0) < binwidth * df

    return mask


# }}}


# {{{ Welch


def evaluate_power_spectrum_density(
    x: Array1D[np.floating[Any]],
    *,
    nwindows: int = 6,
    window_length: int | None = None,
    overlap: float = 0.5,
    nfft: int | None = None,
    fs: float = 1.0,
    p: Any = None,
) -> PowerSpectrumDensity:
    """Evaluate *nwindows* power spectrum densities and check their correlation.

    If the resulting correlations are sufficiently large, the signal can be said to
    be periodic or approach a limit cycle.

    :arg x: an array of shape ``(d, n)``, where :math:`d` is the dimension of
        the state space and :math:`n` is the time step count.
    :arg nwindows: number of windows to consider.
    :arg window_length: length of a single window, which should match with the
        time series length when considering the number of windows.
    :arg overlap: overlap (percentage) between the windows.
    :arg nfft: length of the FFT used to compute the PSD in each window.
    :arg fs: sampling frequency.
    :arg p: norm type used to compute the relative error differences.
    """
    # {{{ validate inputs

    if x.ndim != 1:
        raise ValueError(f"unsupported dimension: {x.ndim}")

    if nwindows <= 0:
        raise ValueError(f"'nwindows' cannot be negative: {nwindows}")

    (n,) = x.shape
    if window_length is None:
        window_length = n // (nwindows + 1)

    if window_length <= 0:
        raise ValueError(f"'window_length' should be positive: {window_length}")

    if not 0 < overlap < 1:
        raise ValueError(f"'overlap' should be in (0, 1): {overlap}")

    if nfft is None:
        nfft = window_length

    if nfft <= 0:
        raise ValueError(f"'nfft' should be positive: {nfft}")

    if fs <= 0:
        raise ValueError(f"'fs' frequency should be positive: '{fs}'")

    # }}}

    # {{{ compute approximate PSD over multiple windows

    from scipy.signal import periodogram

    # Compute PSD for each window
    psds = np.empty((nwindows, window_length // 2 + 1))
    for i, (start, end) in enumerate(
        make_windows(n, nwindows, window_length, overlap=overlap)
    ):
        f, psds[i] = periodogram(
            x[start:end],
            fs=fs,
            nfft=nfft,
            window="hann",
            detrend="constant",
        )

    # }}}

    # {{{ compute periodicity measure

    from scipy.signal import find_peaks

    mean_psd = np.mean(psds, axis=0)
    peaks, props = find_peaks(mean_psd, prominence=0.1 * np.max(mean_psd))

    # 1. Compute harmonic energy
    total_energy = np.sum(mean_psd)
    total_energy = 1.0 if total_energy < 1.0e-8 else total_energy

    if peaks.size:
        f0_idx = peaks[np.argmax(props["prominences"])]
        mask = make_harmonic_mask(f, f[f0_idx], binwidth=2)
        harmonic_energy = np.sum(mean_psd[mask])
    else:
        f0_idx = 0
        harmonic_energy = 0.0

    # }}}

    return PowerSpectrumDensity(harmonic_energy, total_energy, f, psds)


def is_limit_cycle_power_spectrum_density(
    x: Array1D[np.floating[Any]] | Array2D[np.floating[Any]],
    *,
    eps: float = 1.0e-3,
) -> bool:
    if x.ndim == 1:
        x = x.reshape(1, -1)

    return all(
        evaluate_power_spectrum_density(x[i]).is_periodic(eps)
        for i in range(x.shape[0])
    )


# }}}


# {{{ Autocorrelation


@dataclass(frozen=True)
class Autocorrelation:
    corr: Array1D[np.floating[Any]]
    peaks: Array1D[np.integer[Any]]

    def is_periodic(self, eps: float = 5.0e-1) -> bool:
        if self.peaks.size == 0:
            return False

        if len(self.peaks) > 2:
            intervals = np.diff(self.peaks)
            jitter = np.std(intervals) / np.mean(intervals)
        else:
            jitter = 0.0

        confidence = self.corr[self.peaks[0]]
        log.info(
            "Periodic: peaks %.8e (eps %.8e) jitter %.8e (eps %.8e)",
            confidence,
            eps,
            jitter,
            eps,
        )

        return confidence > eps and jitter < eps


def evaluate_auto_correlation(
    x: Array1D[np.floating[Any]],
    *,
    eps: float = 5.0e-1,
    prominence: float | None = None,
    distance: int | None = None,
) -> Autocorrelation:
    if x.ndim != 1:
        raise ValueError(f"unsupported dimension: {x.ndim}")

    if eps < 0:
        raise ValueError(f"'eps' must be positive: {eps}")

    if prominence is not None and prominence < 0:
        raise ValueError(f"'prominence' must be non-negative: {prominence}")

    if distance is not None and distance <= 0:
        raise ValueError(f"'distance' must be positive: {distance}")

    from scipy.signal import correlate, find_peaks
    from scipy.stats import zscore

    # 1. normalize
    x = zscore(x, ddof=1)

    # 2. compute the auto-correlation
    corr = correlate(x, x, mode="full")[x.size - 1 :]
    corr /= np.abs(corr[0])

    # 3. find peaks
    if distance is None:
        distance = 10

    if prominence is None:
        prominence = eps

    peaks, _ = find_peaks(corr, prominence=prominence, distance=distance)

    return Autocorrelation(corr=corr, peaks=peaks)


def is_limit_cycle_auto_correlation(
    x: Array1D[np.floating[Any]],
    *,
    eps: float = 5.0e-1,
) -> bool:
    if x.ndim == 1:
        x = x.reshape(1, -1)

    return all(
        evaluate_auto_correlation(x[i]).is_periodic(eps) for i in range(x.shape[0])
    )


# }}}
