# SPDX-FileCopyrightText: 2025 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np
import numpy.linalg as la

from orbitkit.typing import Array
from orbitkit.utils import module_logger

log = module_logger(__name__)


# {{{ Fourier / Power Spectrum Density


class PSDDeltas(NamedTuple):
    deltas: Array
    """The relative errors between the power spectrum densities computed for
    multiple windows."""
    freq: Array
    """Frequencies of the last computed power spectrum density."""
    psd: Array
    """The last computed power spectrum density."""


def evaluate_welch_power_spectrum_density_deltas(
    x: Array,
    *,
    nwindows: int = 2,
    window_length: int | None = None,
    overlap: float = 0.5,
    nfft: int | None = None,
    fs: float = 1.0,
    p: Any = None,
) -> PSDDeltas:
    """Evaluate *nwindows* power spectrum densities using the Welch algorithm
    and compute their relative difference.

    If the resulting errors are sufficiently small, the signal can be said to
    be periodic, or approach a limit cycle.

    :arg nwindows: number of windows to consider.
    :arg window_length: length of a single window, which should match with the
        time series length when considering the number of windows.
    :arg overlap: overlap (percentage) between the windows.
    :arg nfft: length of the FFT used to compute the PSD in each window.
    :arg fs: sampling frequency.
    :arg p: norm type used to compute the relative error differences.
    """
    # {{{ validate inputs

    if not 0 < overlap < 1:
        raise ValueError(f"'overlap' should be in (0, 1): {overlap}")

    _, n = x.shape
    if window_length is None:
        window_length = n // (nwindows + 1)

    if window_length <= 0:
        raise ValueError(f"'window_length' should be positive: {window_length}")

    if nfft is None:
        nfft = window_length

    if nfft <= 0:
        raise ValueError(f"'nfft' should be positive: {nfft}")

    # }}}

    # {{{ compute approximate PSD using the Welch algorithm over multiple windows

    step = int((1 - overlap) * window_length)
    w_ends = [n - i * step for i in reversed(range(nwindows))]
    w_starts = [max(0, e - window_length) for e in w_ends]

    from scipy.signal import welch

    # Compute PSD for each variable and average
    psds = []
    for start, end in zip(w_starts, w_ends, strict=True):
        f, pxx = welch(
            x[:, start:end],
            fs=fs,
            nperseg=window_length,
            noverlap=int(window_length * overlap),
            nfft=nfft,
            axis=-1,
        )
        psds.append(np.mean(pxx, axis=0))

    # }}}

    # relative difference
    deltas = np.array([
        la.norm(psds[k + 1] - psds[k], ord=p) / la.norm(psds[k], ord=p)
        for k in range(nwindows - 1)
    ])

    return PSDDeltas(deltas, f, psds[-1])


def is_limit_cycle_welch(
    x: Array,
    *,
    eps: float = 1.0e-3,
) -> bool:
    result = evaluate_welch_power_spectrum_density_deltas(x)
    return bool(np.max(result.deltas) < eps)


def _make_lomb_scargle_frequencies(
    t: Array,
    *,
    gamma: float = 5.0,
    fmin: float | None = None,
    fmax: float | None = None,
) -> Array:
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
    t: Array,
    x: Array,
    *,
    nwindows: int = 2,
    window_length: int | None = None,
    overlap: float = 0.5,
    p: Any = None,
) -> PSDDeltas:
    """Evaluate *nwindows* power spectrum densities using the Lomb-Scargle algorithm
    and compute their relative difference.

    If the resulting errors are sufficiently small, the signal can be said to
    be periodic, or approach a limit cycle.

    :arg nwindows: number of windows to consider.
    :arg window_length: length of a single window, which should match with the
        time series length when considering the number of windows.
    :arg overlap: overlap (percentage) between the windows.
    :arg nfreqs: number of frequencies to compute the PSD at.
    :arg p: norm type used to compute the relative error differences.
    """
    # {{{ validate inputs

    d, n = x.shape
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

    # {{{ compute approximate PSD using the Welch algorithm over multiple windows

    # determine frequencies
    freqs = _make_lomb_scargle_frequencies(t)

    # determine segments of `window_length`
    step = int((1 - overlap) * window_length)
    w_ends = [n - i * step for i in reversed(range(nwindows))]
    w_starts = [max(0, e - window_length) for e in w_ends]

    from scipy.signal import lombscargle

    # compute PSD for each variable and average
    psds = []
    for start, end in zip(w_starts, w_ends, strict=True):
        pxx = np.array([
            lombscargle(t[start:end], x[i, start:end], freqs) for i in range(d)
        ])
        psds.append(np.mean(pxx, axis=0))

    # }}}

    # relative difference
    deltas = np.array([
        np.linalg.norm(psds[k + 1] - psds[k], ord=p) / np.linalg.norm(psds[k], ord=p)
        for k in range(nwindows - 1)
    ])

    return PSDDeltas(deltas, freqs, psds[-1])


def is_limit_cycle_lomb_scargle(
    x: Array,
    *,
    eps: float = 1.0e-3,
) -> bool:
    result = evaluate_lomb_scargle_power_spectrum_density_deltas(x)
    return bool(np.max(result.deltas) < eps)


# }}}
