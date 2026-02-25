# SPDX-FileCopyrightText: 2025 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-FileCopyrightText: optype authors
# SPDX-License-Identifier: MIT AND BSD-3-Clause

from __future__ import annotations

import os
from dataclasses import Field
from typing import Any, ClassVar, Protocol, TypeAlias

import numpy as np
from typing_extensions import TypeAliasType, TypeVar

T = TypeVar("T")
"""An unbound invariant generic type variable."""

PathLike: TypeAlias = os.PathLike[str] | str
"""A union of types supported as paths."""

# TODO: this is deprecated, use ArrayND
Array: TypeAlias = np.ndarray[tuple[int, ...], np.dtype[Any]]
"""Array type alias for :class:`numpy.ndarray`."""

# TODO: Should probably just depend on `optype`. We already pull it in through
# `scipy-stubs`. For now this is a very minimalist copy-paste from there. It
# contains parts of
#   optype/numpy/_array.py
#   optype/numpy/_scalar.py

ShapeT = TypeVar("ShapeT", bound=tuple[int, ...], default=tuple[Any, ...])
ScalarTypeT = TypeVar("ScalarTypeT", bound=np.generic, default=Any)

ArrayND = TypeAliasType(
    "ArrayND",
    np.ndarray[ShapeT, np.dtype[ScalarTypeT]],
    type_params=(ShapeT, ScalarTypeT),
)
"""A type alias for a shape and type generic :class:`numpy.ndarray`."""

Array0D = TypeAliasType(
    "Array0D",
    np.ndarray[tuple[()], np.dtype[ScalarTypeT]],
    type_params=(ScalarTypeT,),
)
"""A type alias for a 0-dimensional :class:`ArrayND`."""

Array1D = TypeAliasType(
    "Array0D",
    np.ndarray[tuple[int], np.dtype[ScalarTypeT]],
    type_params=(ScalarTypeT,),
)
"""A type alias for a 1-dimensional :class:`ArrayND`."""

Array2D = TypeAliasType(
    "Array0D",
    np.ndarray[tuple[int, int], np.dtype[ScalarTypeT]],
    type_params=(ScalarTypeT,),
)
"""A type alias for a 2-dimensional :class:`ArrayND`."""

Array3D = TypeAliasType(
    "Array0D",
    np.ndarray[tuple[int, int, int], np.dtype[ScalarTypeT]],
    type_params=(ScalarTypeT,),
)

# TODO: we probably want to also support complex "scalars" down the road

Float: TypeAlias = float | np.floating[Any]
"""Type alias for admissible float types."""
Scalar: TypeAlias = int | float | np.floating[Any]
"""Scalar type alias (generally a value convertible to a :class:`float`)."""
ScalarLike: TypeAlias = Scalar | Array0D[np.number[Any]]
"""A scalar-like value, which may include array of shape ``()``."""

InexactT = TypeVar("InexactT", bound=np.inexact[Any])


class DataclassInstance(Protocol):
    """Dataclass protocol from
    `typeshed <https://github.com/python/typeshed/blob/770724013de34af6f75fa444cdbb76d187b41875/stdlib/_typeshed/__init__.pyi#L329-L334>`__."""

    __dataclass_fields__: ClassVar[dict[str, Field[Any]]]


DataclassInstanceT = TypeVar("DataclassInstanceT", bound=DataclassInstance)
"""An invariant :class:`~typing.TypeVar` bound to :class:`DataclassInstance`."""
