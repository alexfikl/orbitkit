# SPDX-FileCopyrightText: 2025 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from orbitkit.utils import enable_test_plotting, module_logger
from orbitkit.visualization import figure, set_plotting_defaults

TEST_FILENAME = pathlib.Path(__file__)
TEST_DIRECTORY = TEST_FILENAME.parent

log = module_logger(__name__)
set_plotting_defaults()


# {{{ test_detect_cycle_harmonic


@pytest.mark.parametrize("b", [0.1, 0.01, 0.001])
def test_detect_cycle_harmonic(b: float) -> None:
    rng = np.random.default_rng(seed=32)

    # generate some random data with some noise
    a = -0.001
    theta = np.linspace(0.0, 32 * np.pi, 2048)
    x = np.sin(theta) * np.exp(-a * theta) + b * rng.normal(size=theta.shape)

    from orbitkit.cycles import detect_cycle_harmonic

    result = detect_cycle_harmonic(x, nwindows=6)

    error = 1.0 - result.harmonic_energy / result.total_energy
    log.info("Error: %.8e (%.8e)", error, b)
    assert error < 15.0 * b

    if not enable_test_plotting():
        return

    with figure(
        TEST_DIRECTORY / "test_detect_cycle_harmonic_psd", normalize=True
    ) as fig:
        ax = fig.gca()

        ax.plot(result.freq, result.psd.T)
        ax.set_xlabel("$k$")
        ax.set_ylabel("PSD")


# }}}


# {{{ test_make_harmonic_mask


def test_make_harmonic_mask_bins() -> None:
    """Correct frequency bins are masked for each harmonic."""
    from orbitkit.cycles import make_harmonic_mask

    nfft = 1024
    f = np.fft.rfftfreq(nfft)
    df = f[1] - f[0]
    f0 = 0.1

    mask = make_harmonic_mask(f, f0, nharmonics=3, binwidth=1)

    # mean must not be masked
    assert not mask[0]

    # each of the first 3 harmonics must have at least one bin masked
    for k in range(1, 4):
        expected_bin = round(k * f0 / df)
        assert mask[expected_bin], f"harmonic {k} (bin {expected_bin}) not masked"

    # bins well away from any harmonic must not be masked
    off_bin = round(0.055 / df)
    assert not mask[off_bin]


def test_make_harmonic_mask_binwidth() -> None:
    """binwidth controls the number of bins captured on each side."""
    from orbitkit.cycles import make_harmonic_mask

    nfft = 512
    f = np.fft.rfftfreq(nfft)
    df = f[1] - f[0]
    f0 = 0.1
    h1_bin = round(f0 / df)

    mask_narrow = make_harmonic_mask(f, f0, nharmonics=1, binwidth=1)
    mask_wide = make_harmonic_mask(f, f0, nharmonics=1, binwidth=4)

    assert mask_narrow.sum() <= mask_wide.sum()
    # with binwidth=1 the mask covers at most 2 bins per harmonic (centered ±1)
    assert mask_narrow.sum() <= 2

    # immediate neighbours of the centre bin are in the wide mask
    assert mask_wide[h1_bin - 2]
    assert mask_wide[h1_bin + 2]


# }}}


# {{{ test_detect_cycle_harmonic_is_periodic


def test_detect_cycle_harmonic_is_periodic() -> None:
    """is_periodic() returns True for a clean sinusoidal signal."""
    from orbitkit.cycles import detect_cycle_harmonic

    theta = np.linspace(0.0, 32 * np.pi, 2048)
    x = np.sin(theta)

    result = detect_cycle_harmonic(x, nwindows=6)
    assert result.is_periodic(), f"expected periodic, error={result.error}"


# }}}


# {{{ test_detect_cycle_harmonic_nfft


def test_detect_cycle_harmonic_nfft_shape() -> None:
    """psd and freq shapes match nfft regardless of window_length."""
    from orbitkit.cycles import detect_cycle_harmonic

    x = np.sin(np.linspace(0.0, 32 * np.pi, 2048))

    for nfft in [128, 256, 512]:
        result = detect_cycle_harmonic(x, nwindows=6, nfft=nfft)
        expected_bins = nfft // 2 + 1
        assert result.freq.shape == (expected_bins,)
        assert result.psd.shape[1] == expected_bins


# }}}


# {{{ test_detect_cycle_harmonic_zero_signal


def test_detect_cycle_harmonic_zero_signal() -> None:
    """All-zero input is handled gracefully and is not declared periodic."""
    from orbitkit.cycles import detect_cycle_harmonic

    x = np.zeros(2048)
    result = detect_cycle_harmonic(x, nwindows=6)
    assert result.is_periodic()


# }}}


# {{{ test_detect_cycle_harmonic_short_signal


def test_detect_cycle_harmonic_short_signal() -> None:
    """Signals too short to fill all windows raise ValueError."""
    from orbitkit.cycles import detect_cycle_harmonic

    # 50 samples with window_length=20: required = 20 + 5*10 = 70 > 50
    x = np.sin(np.linspace(0.0, 2 * np.pi, 50))
    with pytest.raises(ValueError, match="signal length too small"):
        detect_cycle_harmonic(x, nwindows=6, window_length=20)


# }}}


# {{{ test_detect_cycle_harmonic_aperiodic


def test_detect_cycle_harmonic_aperiodic() -> None:
    """White noise is not declared periodic."""
    from orbitkit.cycles import detect_cycle_harmonic

    rng = np.random.default_rng(seed=42)
    x = rng.standard_normal(4096)

    result = detect_cycle_harmonic(x, nwindows=6)
    assert not result.is_periodic(), result.error


# }}}


# {{{ test_detect_cycle_harmonic_nonlinear


def test_detect_cycle_harmonic_nonlinear() -> None:
    """A nonlinear limit cycle (sum of harmonics) is detected as periodic."""
    from orbitkit.cycles import detect_cycle_harmonic

    n = 8192
    theta = np.linspace(0.0, 128 * np.pi, n)
    x = np.sin(theta) + 0.3 * np.sin(2 * theta) + 0.1 * np.sin(3 * theta)

    result = detect_cycle_harmonic(x, nwindows=6)
    assert result.is_periodic(), result.error


# }}}


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])
