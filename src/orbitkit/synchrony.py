# SPDX-FileCopyrightText: 2025 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import numpy as np

from orbitkit.typing import Array
from orbitkit.utils import module_logger

log = module_logger(__name__)


def pfeuty_chi(V: Array, *, ddof: int = 1) -> float:
    r"""Computes the synchrony measure from [Pfeuty2007]_.

    This measure is based on the normalized standard deviation of the membrane
    potential. As stated in [Pfeuty2007]_, the network is considered to be
    synchronized if :math:`\chi(N) \sim 1` and asynchronous if :math:`\chi(N) < 1`.

    Note that this assumes that the samples in :math:`V` are uniformly spaced.

    .. [Pfeuty2007] B. Pfeuty,
        *Inhibition Potentiates the Synchronizing Action of Electrical Synapses*,
        Frontiers in Computational Neuroscience, Vol. 1, 2007,
        `doi:10.3389/neuro.10.008.2007 <https://doi.org/10.3389/neuro.10.008.2007>`__.

    :arg V: an array of shape ``(nnodes, ntimesteps)`` for which to compute the
        synchrony measure.
    """

    # Equation 8+9: compute variance of mean
    Vhat = np.mean(V, axis=0)
    varhat = np.var(Vhat, ddof=ddof)

    # Equation 10: compute variance component-wise
    var = np.var(V, ddof=1, axis=ddof)
    assert var.shape == (V.shape[0],)

    # Equation 11: compute chi
    chi = np.sqrt(varhat / np.mean(var))

    return chi  # type: ignore[no-any-return]


def kuramoto_order_parameters(V: Array) -> Array:
    r"""Compute Kuramoto's order parameter for the time series *V*.

    The order parameter is obtained by first taking the Hilbert transform of the
    signal and using the corresponding argument as the phase in the Kuramoto
    model. Then the order parameter :math:`r(t)` is computed as usual

    .. math::

        r^n e^{\imath \psi^n} = \frac{1}{n} \sum_{j = 0}^N e^{\imath \theta_j^n}.

    where :math:`n` denotes the time step and :math:`N` is the number of nodes.

    :arg V: and array of shape ``(nnodes, ntimesteps)`` describing the signal.
    :returns: an array of size ``(ntimesteps,)`` containing the order parameter
        :math:`r(t)`.
    """

    from scipy.signal import hilbert

    # get angle of signal
    Vhat = hilbert(V, axis=1)
    theta = np.angle(Vhat)

    # compute order parameter
    r = np.abs(np.mean(np.exp(1j * theta), axis=0))

    return r  # type: ignore[no-any-return]


def kuramoto_order_parameter(V: Array, *, method: str = "hilbert") -> float:
    """Compute an average of the Kuramoto order parameter.

    See :func:`kuramoto_order_parameters` to get the value at each time step.
    """
    return float(np.mean(kuramoto_order_parameters(V)))


def kuramoto_order_parameter_network(V: Array, *, mat: Array) -> Array:
    raise NotImplementedError


def kuramoto_order_parameter_mean_field(V: Array, *, mat: Array) -> Array:
    raise NotImplementedError


def kuramoto_order_parameter_link(V: Array, *, mat: Array) -> Array:
    raise NotImplementedError


def kuramoto_order_parameter_universal(V: Array, *, mat: Array) -> Array:
    # http://dx.doi.org/10.1063/1.4995963
    raise NotImplementedError
