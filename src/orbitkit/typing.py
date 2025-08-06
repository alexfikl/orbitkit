# SPDX-FileCopyrightText: 2025 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import os
from typing import Any, TypeAlias, TypeVar

import numpy as np

T = TypeVar("T")
"""An unbound invariant generic type variable."""

PathLike: TypeAlias = os.PathLike[str] | str
"""A union of types supported as paths."""

Array: TypeAlias = np.ndarray[tuple[int, ...], np.dtype[Any]]
"""Array type alias for :class:`numpy.ndarray`."""
Float: TypeAlias = float | np.floating[Any]
"""Type alias for admissible float types."""
Scalar: TypeAlias = int | float | np.number[Any]
"""Scalar type alias (generally a value convertible to a :class:`float`)."""
ScalarLike: TypeAlias = Scalar | np.ndarray[tuple[int], np.dtype[np.generic]]
"""A scalar-like value, which may include array of shape ``()``."""
