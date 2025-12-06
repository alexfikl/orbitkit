# SPDX-FileCopyrightText: 2025 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, fields, is_dataclass, replace
from typing import Any

import numpy as np

import orbitkit.symbolic.primitives as sym
from orbitkit.typing import DataclassInstanceT
from orbitkit.utils import module_logger

log = module_logger(__name__)


# {{{ dataclass as symbolic


def ds_symbolic(
    obj: DataclassInstanceT,
    *,
    rec: bool = False,
    rattrs: set[str] | None = None,
) -> DataclassInstanceT:
    """Fill in all the fields of *cls* with symbolic variables.

    :arg rec: if *True*, automatically recurse into all child dataclasses.
    :arg rattrs: a set of attribute names that will be recursed into regardless
        of the value of the *rec* flag.
    """

    if rattrs is None:
        rattrs = set()

    kwargs: dict[str, Any] = {}
    for f in fields(obj):
        attr = getattr(obj, f.name)
        if (rec or f.name in rattrs) and is_dataclass(attr):
            assert not isinstance(attr, type)
            kwargs[f.name] = ds_symbolic(attr, rec=rec, rattrs=rattrs)
            continue

        if isinstance(attr, tuple):
            kwargs[f.name] = tuple(
                sym.Variable(f"{f.name}_{i}") for i in range(len(attr))
            )
        elif isinstance(attr, np.ndarray):
            kwargs[f.name] = sym.MatrixSymbol(f.name, attr.shape)
        else:
            kwargs[f.name] = sym.Variable(f.name)

    return replace(obj, **kwargs)


# }}}


# {{{ model


@dataclass(frozen=True)
class Model(ABC):
    @property
    @abstractmethod
    def variables(self) -> tuple[str, ...]:
        """A tuple of all the state variables in the system."""

    @abstractmethod
    def evaluate(
        self, t: sym.Expression, *args: sym.MatrixSymbol
    ) -> tuple[sym.Expression, ...]:
        """
        :returns: an expression of the model evaluated at the given arguments.
        """

    def symbolify(
        self,
        n: int | tuple[int, ...],
        *,
        full: bool = False,
    ) -> tuple[tuple[sym.Variable, ...], tuple[sym.Expression, ...]]:
        r"""Evaluate model on symbolic arguments for a specific size *n*.

        This function creates appropriate symbolic variables and calls
        :meth:`evaluate` to create the fully symbolic model. These can then
        essentially be passed directly to some other backend for code generation.

        :returns: a tuple of ``(args, model)``, where *args* are the symbolic
            variables (i.e. :class:`~pymbolic.primitives.Variable` and such) and
            the model is given as a tuple of symbolic expression (one for each
            input variable).
        """

        x = self.variables
        if isinstance(n, int):
            n = (n,) * len(x)

        if len(x) != len(n):
            raise ValueError(
                f"number of variables does not match sizes: variables {x} for sizes {n}"
            )

        if not all(n[0] == n_i for n_i in n[1:]):
            raise NotImplementedError(f"only uniform sizes are supported: {n}")

        t = sym.Variable("t")
        args = [sym.MatrixSymbol(name, (n_i,)) for n_i, name in zip(n, x, strict=True)]

        model = self
        if full:
            model = ds_symbolic(model, rec=False, rattrs={"param"})

        return (t, *args), model.evaluate(t, *args)

    def __str__(self) -> str:
        n = getattr(self, "n", None)
        if n is None:
            n = 8

        from orbitkit.symbolic.mappers import stringify

        args, exprs = self.symbolify(n, full=True)

        eqs = []
        for i, (name, eq) in enumerate(zip(args[1:], exprs, strict=True)):
            eqs.append(f"[{i:02d}]:\n\td{stringify(name)}/dt = {stringify(eq)}")

        return "\n".join(eqs)


# }}}
