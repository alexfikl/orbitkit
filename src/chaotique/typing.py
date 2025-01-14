# SPDX-FileCopyrightText: 2025 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import numpy as np

PathLike = os.PathLike[str] | str
"""A union of types supported as paths."""

if TYPE_CHECKING:
    Array = np.ndarray[Any, np.dtype[Any]]
    Scalar = np.number[Any] | Array
else:
    Array = np.ndarray
    """Array type alias for :class:`numpy.ndarray`."""
    Scalar = np.number | Array
    """Scalar type alias (generally a value convertible to a :class:`float`)."""
