# SPDX-FileCopyrightText: 2025 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import os
from dataclasses import Field
from typing import Any, ClassVar, Protocol, TypeAlias, TypeVar

import numpy as np

T = TypeVar("T")
"""An unbound invariant generic type variable."""

PathLike: TypeAlias = os.PathLike[str] | str
"""A union of types supported as paths."""

Array: TypeAlias = np.ndarray[tuple[int, ...], np.dtype[Any]]
"""Array type alias for :class:`numpy.ndarray`."""
Float: TypeAlias = float | np.floating[Any]
"""Type alias for admissible float types."""
Scalar: TypeAlias = int | float | np.floating[Any]
"""Scalar type alias (generally a value convertible to a :class:`float`)."""
ScalarLike: TypeAlias = Scalar | np.ndarray[tuple[int], np.dtype[np.generic]]
"""A scalar-like value, which may include array of shape ``()``."""


class DataclassInstance(Protocol):
    """Dataclass protocol from
    `typeshed <https://github.com/python/typeshed/blob/770724013de34af6f75fa444cdbb76d187b41875/stdlib/_typeshed/__init__.pyi#L329-L334>`__."""

    __dataclass_fields__: ClassVar[dict[str, Field[Any]]]


DataclassInstanceT = TypeVar("DataclassInstanceT", bound=DataclassInstance)
"""An invariant :class:`~typing.TypeVar` bound to :class:`DataclassInstance`."""
