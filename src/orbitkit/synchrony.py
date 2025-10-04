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


def kuramoto_order_parameter(theta: Array) -> Array:
    r"""Compute Kuramoto's order parameter for the time series *theta*.

    Then the order parameter :math:`r(t)` is computed as usual (see e.g.
    [Schroder2017]_):

    .. math::

        r(t) e^{\imath \psi(t)} = \frac{1}{n} \sum_{j = 0}^n e^{\imath \theta_j(t)},

    where :math:`N` is the number of nodes in the network. Explicitly, the order
    parameter cam be written as

    .. [Schroder2017] M. Schröder, M. Timme, D. Witthaut,
        *A Universal Order Parameter for Synchrony in Networks of
        Limit Cycle Oscillators*,
        Chaos: An Interdisciplinary Journal of Nonlinear Science, Vol. 27, 2017,
        `doi:10.1063/1.4995963 <https://doi.org/10.1063/1.4995963>`__.

    :arg theta: and array of shape ``(nnodes, ntimesteps)`` describing the signal.
    :returns: an array of size ``(ntimesteps,)`` containing the order parameter
        :math:`r(t)`.
    """
    return np.abs(np.mean(np.exp(1j * theta), axis=0))  # type: ignore[no-any-return]


def global_kuramoto_order_parameter(theta: Array) -> float:
    r"""Compute an average of the classic Kuramoto order parameter.

    See :func:`kuramoto_order_parameter` to get the value at each time step. The
    global order parameter is then given by

    .. math::

        r_{\text{classic}} = \langle r(t) \rangle_t.
    """
    return float(np.mean(kuramoto_order_parameter(theta), axis=-1))


def global_kuramoto_order_parameter_network(theta: Array, mat: Array) -> float:
    r"""Compute a network-sensitive order parameter (see Equation 6 in [Schroder2017]_).

    This order parameter is given by

    .. math::

        r_{\text{net}} = \frac{\sum_{i = 0}^{n - 1} r_i}
                              {\sum_{i = 0}^{n - 1} k_i},

    where

    .. math::

        r_i = \sum_{j = 0}^{n - 1} A_{i j} \langle e^{\imath \theta_j(t)} \rangle_t.

    :arg mat: a binary adjacency matrix for the system.
    :returns: an array of shape ``(nnodes, ntimesteps)`` containing the order
        parameters :math:`r_i(t)`.
    """
    r"""Compute an average of the network Kuramoto order parameter.


    See :func:`kuramoto_order_parameter_network` to get the value at each time step.
    """
    n, _ = theta.shape
    if mat.shape != (n, n):
        raise ValueError(
            "adjacency matrix 'mat' size does not match 'theta': "
            f"matrix has shape {mat.shape} and theta has shape {theta.shape}"
        )

    r = np.abs(mat @ np.mean(np.exp(1j * theta), axis=-1))
    k = np.sum(mat != 0, axis=1)

    return float(np.sum(r) / np.sum(k))


def kuramoto_order_parameter_mean_field(theta: Array, mat: Array) -> Array:
    r"""Compute a network-sensitive mean-field order parameter (see Equation 7
    in [Schroder2017]_).

    .. math::

        r(t) e^{\imath \psi(t)} =
            \frac{\sum k_i e^{\imath \theta_i(t)}}{\sum k_i},

    where :math:`k_i` is the degree of the node :math:`i`. Note that, unlike
    :func:`kuramoto_order_parameter_network`, this measure uses an average
    over all the node neighbors (i.e. a "mean field" approach).

    :arg mat: a binary adjacency matrix for the system.
    :returns: an array of size ``(ntimesteps,)`` containing the order parameter
        :math:`r(t)`.
    """

    n, _ = theta.shape
    if mat.shape != (n, n):
        raise ValueError(
            "adjacency matrix 'mat' size does not match 'theta': "
            f"matrix has shape {mat.shape} and theta has shape {theta.shape}"
        )

    k = np.sum(mat != 0, axis=1).reshape(-1, 1)
    return np.abs(np.sum(k * np.exp(1j * theta), axis=0) / np.sum(k))  # type: ignore[no-any-return]


def global_kuramoto_order_parameter_mean_field(theta: Array, mat: Array) -> float:
    """Compute an average of the mean field Kuramoto order parameter.

    See :func:`kuramoto_order_parameter_mean_field` to get the value at each time step.
    """
    return float(np.mean(kuramoto_order_parameter_mean_field(theta, mat), axis=-1))


def global_kuramoto_order_parameter_link(theta: Array, mat: Array) -> float:
    r"""Compute a network-sensitive order parameter (see Equation 8 in [Schroder2017]_).

    This order parameter is given by

    .. math::

        r_{\text{link}} = \frac{1}{\sum k_i} \sum_{i, j = 0}^{n - 1}
            A_{ij} |\langle e^{\imath (\theta_i - \theta_j)} \rangle_t|,

    where :math:`k_i` is the degree of node :math:`i`.
    """
    n, _ = theta.shape
    if mat.shape != (n, n):
        raise ValueError(
            "adjacency matrix 'mat' size does not match 'theta': "
            f"matrix has shape {mat.shape} and theta has shape {theta.shape}"
        )

    k = np.sum(mat != 0, axis=1)
    dt = np.mean(np.exp(1j * (theta[None, :, :] - theta[:, None, :])), axis=-1)

    return float(np.sum(mat * np.abs(dt)) / np.sum(k))


def global_kuramoto_order_parameter_universal(theta: Array, mat: Array) -> float:
    r"""Compute a universal network-sensitive order parameter (see Equation 9
    from [Schroder2017]_).

    This order parameter is given by:

    .. math::

        r_{\text{uni}} = \frac{1}{\sum k_i} \sum_{i, j = 0}^{n - 1}
            A_{ij} \langle \cos (\theta_i - \theta_j)\rangle_t

    where :math:`k_i` is the degree of node :math:`i`.
    """

    n, _ = theta.shape
    if mat.shape != (n, n):
        raise ValueError(
            "adjacency matrix 'mat' size does not match 'theta': "
            f"matrix has shape {mat.shape} and theta has shape {theta.shape}"
        )

    k = np.sum(mat != 0, axis=1)
    dcos = np.mean(np.cos(theta[None, :, :] - theta[:, None, :]), axis=-1)

    return float(np.sum(mat * dcos) / np.sum(k))


def kemeth_spatial_correlation_measure(
    V: Array,
    *,
    atol: float | None = None,
    rtol: float = 1.0e-2,
) -> Array:
    r"""Compute the :math:`g_0` correlation measure from [Kemeth2016]_.

    The correlation measure is computed using Equation (4) from [Kemeth2016]_.
    In our notation, *atol* takes the places of :math:`\delta` as the desired
    threshold. If *atol* is not given, then *rtol* is used to compute it
    using :math:`\epsilon_{\text{rel}} D_m`, where :math:`D_m` is the maximum
    pairwise distance.

    In practice, the correlation is given by

    .. math::

        g_0 = \sqrt{
            \frac{\#\{(i, j) \mid i < j, D_{ij} < \epsilon_{\text{abs}}\}}{\binom{n}{2}}
            }

    .. [Kemeth2016]
        F. P. Kemeth, S. W. Haugland, L. Schmidt, I. G. Kevrekidis, K. Krischer,
        *A Classification Scheme for Chimera States*,
        Chaos: An Interdisciplinary Journal of Nonlinear Science, Vol. 26, 2016,
        `doi:10.1063/1.4959804 <https://doi.org/10.1063/1.4959804>`__.
    """
    if V.ndim == 1:
        V = V.reshape(-1, 1)

    if V.ndim != 2:
        raise ValueError(f"only 2d time series are supported: {V.shape}")

    if not 0.0 < rtol <= 1.0:
        raise ValueError(f"'rtol' must be in (0, 1]: {rtol}")

    if atol is not None and atol <= 0.0:
        raise ValueError(f"'atol' must be non-negative: {atol}")

    d, n = V.shape
    if d < 2:
        return np.full(V.shape[1], 1.0, dtype=V.real.dtype)

    # compute pairwise distances at each time step
    dists = np.abs(V[:, None, :] - V[None, :, :])
    # select only distances for i < j
    dists = dists[np.triu_indices(n, k=1)]

    dmax = np.max(dists)
    if atol is None:
        atol = max(rtol * dmax, np.finfo(V.dtype).eps)

    g0 = np.empty(n, dtype=dists.dtype)
    for i in range(n):
        g0[i] = np.sqrt(np.mean(dists[:, i] < atol))

    return g0
