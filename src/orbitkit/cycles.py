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
    deltas: Array1D[np.floating[Any]]
    """Pairwise correlations between the PSDs of each window."""

    freq: Array1D[np.floating[Any]]
    """Frequencies of the last computed power spectrum density."""
    psd: Array1D[np.floating[Any]]
    """The last computed power spectrum density."""

    def is_periodic(self, eps: float = 1.0e-3) -> bool:
        """Check if the corresponding time series is periodic based on the PSD."""
        delta = np.min(1 - self.deltas)

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


def evaluate_welch_power_spectrum_density_deltas(
    x: Array1D[np.floating[Any]],
    *,
    nwindows: int = 6,
    window_length: int | None = None,
    overlap: float = 0.5,
    nfft: int | None = None,
    fs: float = 1.0,
    p: Any = None,
) -> PowerSpectrumDensity:
    """Evaluate *nwindows* power spectrum densities using `Welch's method
    <https://en.wikipedia.org/wiki/Welch%27s_method>`__ and check their
    correlation.

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

    # {{{ compute approximate PSD using the Welch algorithm over multiple windows

    from scipy.signal import periodogram

    # Compute PSD for each window
    psds = []
    for start, end in make_windows(n, nwindows, window_length, overlap=overlap):
        f, pxx = periodogram(
            x[start:end],
            fs=fs,
            nfft=nfft,
            window="hann",
            detrend="constant",
        )
        psds.append(pxx)

    # }}}

    # compute correlations
    from scipy.stats import pearsonr

    # NOTE: we cannot look at a norm of the PSD because the windowing makes for
    # uneven spectral leakage that will result in large errors. Some correlation,
    # like Pearson, seems to work a lot more reliably.
    deltas = np.array([pearsonr(psds[k + 1], psds[k])[0] for k in range(nwindows - 1)])

    return PowerSpectrumDensity(deltas, f, pxx)


def is_limit_cycle_welch(
    x: Array1D[np.floating[Any]] | Array2D[np.floating[Any]],
    *,
    eps: float = 1.0e-3,
) -> bool:
    if x.ndim == 1:
        x = x.reshape(1, -1)

    return all(
        evaluate_welch_power_spectrum_density_deltas(x[i]).is_periodic(eps)
        for i in range(x.shape[0])
    )


# }}}


# {{{ Lomb-Scargle


def _make_lomb_scargle_frequencies(
    t: Array1D[np.floating[Any]],
    *,
    gamma: float = 5.0,
    fmin: float | None = None,
    fmax: float | None = None,
) -> Array1D[np.floating[Any]]:
    r"""
    :arg gamma: oversampling factor.
    :arg fmin: minimum considered frequency, defaults to :math:`1 / T` (the
        inverse interval size).
    :arg fmax: maximum considered frequency, defaults to :math:`1 / \Delta t / 2`,
        where :math:`\Delta t` is the median interval size.
    """
    # NOTE: assume that the array is sorted
    T = t[-1] - t[0]

    # determine frequency bounds
    if fmin is None:
        fmin = 1.0 / T

    if fmax is None:
        dt = np.median(np.diff(t))
        fmax = 1.0 / (2.0 * dt)

    # determine angular frequency bounds
    wmin = 2.0 * np.pi * fmin
    wmax = 2.0 * np.pi * fmax

    # get number of points
    n = max(128, int(gamma * (fmax - fmin) * T))

    return np.linspace(wmin, wmax, n)


def evaluate_lomb_scargle_power_spectrum_density_deltas(
    t: Array1D[np.floating[Any]],
    x: Array1D[np.floating[Any]] | Array2D[np.floating[Any]],
    *,
    nwindows: int = 6,
    window_length: int | None = None,
    overlap: float = 0.5,
    p: Any = None,
) -> PowerSpectrumDensity:
    """Evaluate *nwindows* power spectrum densities using the Lomb-Scargle algorithm
    and compute their relative difference.

    The main difference between this function and
    :func:`evaluate_welch_power_spectrum_density_deltas` is support for non-uniform
    spaced samples. Note that this will make the function slower, as expected, so
    it may be better to interpolate the data to a uniform grid instead.

    :arg nwindows: number of windows to consider.
    :arg window_length: length of a single window, which should match with the
        time series length when considering the number of windows.
    :arg overlap: overlap (percentage) between the windows.
    :arg nfreqs: number of frequencies to compute the PSD at.
    :arg p: norm type used to compute the relative error differences.
    """
    # {{{ validate inputs

    if x.ndim != 1:
        raise ValueError(f"unsupported dimension: {x.ndim}")

    (n,) = x.shape
    if t.shape != (n,):
        raise ValueError(
            f"array sizes do not match: 't' has size {t.size} (expected {n})"
        )

    if not 0 < overlap < 1:
        raise ValueError(f"'overlap' should be in (0, 1): {overlap}")

    if window_length is None:
        window_length = n // (nwindows + 1)

    if window_length <= 0:
        raise ValueError(f"'window_length' should be positive: {window_length}")

    # }}}

    # {{{ compute approximate PSD using the Lomb-Scargle algorithm over multiple windows

    from scipy.signal import lombscargle

    # determine frequencies
    freqs = _make_lomb_scargle_frequencies(t)

    # compute PSD for each window
    psds = []
    for start, end in make_windows(n, nwindows, window_length, overlap=overlap):
        pxx = lombscargle(t[start:end], x[start:end], freqs)
        psds.append(pxx)

    # }}}

    # compute correlations
    from scipy.stats import pearsonr

    # NOTE: we cannot look at a norm of the PSD because the windowing makes for
    # uneven spectral leakage that will result in large errors. Some correlation,
    # like Pearson, seems to work a lot more reliably.
    deltas = np.array([pearsonr(psds[k + 1], psds[k])[0] for k in range(nwindows - 1)])

    return PowerSpectrumDensity(deltas, freqs, psds[-1])


def is_limit_cycle_lomb_scargle(
    t: Array1D[np.floating[Any]],
    x: Array1D[np.floating[Any]],
    *,
    eps: float = 1.0e-3,
) -> bool:
    if x.ndim == 1:
        x = x.reshape(1, -1)

    return all(
        evaluate_lomb_scargle_power_spectrum_density_deltas(t, x[i]).is_periodic(eps)
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
