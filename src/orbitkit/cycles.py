# SPDX-FileCopyrightText: 2025 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from orbitkit.typing import Array
from orbitkit.utils import module_logger

log = module_logger(__name__)


# {{{ Fourier / Power Spectrum Density


class PSDLimitCycle(NamedTuple):
    is_limit_cycle: bool
    rdiff: float
    freq: Array
    psd: Array


def characterize_limit_cycle_psd(
    x: Array,
    *,
    overlap: float = 0.5,
    window_length: int | None = None,
    nfft: int | None = None,
    fs: float = 1.0,
    eps: float = 1.0e-3,
) -> PSDLimitCycle:
    # {{{ validate inputs

    if eps <= 0:
        raise ValueError(f"'eps' should be positive: {eps}")

    if not 0 < overlap < 1:
        raise ValueError(f"'overlap' should be in (0, 1): {overlap}")

    _, n = x.shape
    if window_length is None:
        window_length = n // 4

    if window_length <= 0:
        raise ValueError(f"'window_length' should be positive: {window_length}")

    if nfft is None:
        nfft = window_length

    if nfft <= 0:
        raise ValueError(f"'nfft' should be positive: {nfft}")

    # }}}

    step = int((1 - overlap) * window_length)
    idx2_end = n
    idx2_start = max(0, idx2_end - window_length)
    idx1_end = idx2_start + step
    idx1_start = max(0, idx1_end - window_length)

    from scipy.signal import welch

    # Compute PSD for each variable and average
    segs = [(idx1_start, idx1_end), (idx2_start, idx2_end)]
    psds = []
    for start, end in segs:
        f, pxx = welch(
            x[:, start:end],
            fs=fs,
            nperseg=window_length,
            noverlap=int(window_length * overlap),
            nfft=nfft,
            axis=-1,
        )
        psds.append(pxx.mean(axis=0))

    # relative difference (L2 norm)
    psd1, psd2 = psds
    rdiff = np.linalg.norm(psd2 - psd1) / np.linalg.norm(psd1)

    return PSDLimitCycle(bool(rdiff < eps), float(rdiff), f, psd2)


def is_limit_cycle_psd(
    x: Array,
    *,
    eps: float = 1.0e-3,
) -> bool:
    result = characterize_limit_cycle_psd(x, eps=eps)
    return result.is_limit_cycle


# }}}
