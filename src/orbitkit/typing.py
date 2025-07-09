# SPDX-FileCopyrightText: 2025 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import os
from typing import Any, TypeVar

import numpy as np

T = TypeVar("T")
"""An unbound invariant generic type variable."""

PathLike = os.PathLike[str] | str
"""A union of types supported as paths."""

Array = np.ndarray[Any, np.dtype[Any]]
"""Array type alias for :class:`numpy.ndarray`."""
Scalar = np.number[Any] | Array
"""Scalar type alias (generally a value convertible to a :class:`float`)."""
