# SPDX-FileCopyrightText: 2025 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, fields, is_dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np

import orbitkit.symbolic.primitives as sym
from orbitkit.typing import DataclassInstanceT
from orbitkit.utils import module_logger

if TYPE_CHECKING:
    from collections.abc import Mapping

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

    def _ds_field_symbolic(  # noqa: PLR0911
        attr: object, fname: str, *, rec: bool, rattrs: set[str]
    ) -> object:
        from pymbolic.primitives import ExpressionNode

        if isinstance(attr, tuple):
            if rec or any(fname.startswith(rattr) for rattr in rattrs):
                return tuple(
                    _ds_field_symbolic(attr[i], f"{fname}_{i}", rec=rec, rattrs=rattrs)
                    for i in range(len(attr))
                )
            else:
                return tuple(sym.Variable(f"{fname}_{i}") for i in range(len(attr)))
        if isinstance(attr, dict):
            if rec or any(fname.startswith(rattr) for rattr in rattrs):
                return {
                    k: _ds_field_symbolic(
                        attr[k],  # ty: ignore[invalid-argument-type]
                        f"{fname}_{k}",
                        rec=rec,
                        rattrs=rattrs,
                    )
                    for k in attr
                }
            else:
                return {k: sym.Variable(f"{fname}_{k}") for k in attr}

        elif isinstance(attr, np.ndarray):
            return sym.MatrixSymbol(fname, attr.shape)
        elif isinstance(attr, (int, float, np.number, ExpressionNode)):
            return attr
        else:
            return sym.Variable(fname)

    kwargs: dict[str, Any] = {}
    for f in fields(obj):
        fname = f.name
        attr = getattr(obj, fname)
        if (rec or fname in rattrs) and is_dataclass(attr):
            assert not isinstance(attr, type)
            kwargs[fname] = ds_symbolic(attr, rec=rec, rattrs=rattrs)
            continue

        kwargs[fname] = _ds_field_symbolic(attr, fname, rec=rec, rattrs=rattrs)

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

    @property
    def rattrs(self) -> set[str]:
        return {"param"}

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
            model = ds_symbolic(model, rec=False, rattrs=self.rattrs)

        from orbitkit.symbolic.mappers import flatten

        return (t, *args), flatten(model.evaluate(t, *args))

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


# {{{ linear_chain_trick


@dataclass(frozen=True)
class ExtendedLinearChainTrickModel(Model):
    orig: Model
    exprs: tuple[sym.Expression, ...]
    equations: Mapping[str, sym.Expression]

    @property
    def n(self) -> int:
        n = getattr(self.orig, "n", None)
        if n is None:
            raise AttributeError("n")

        return n

    @property
    def variables(self) -> tuple[str, ...]:
        return (*self.orig.variables, *self.equations)

    @property
    def rattrs(self) -> set[str]:
        return {*self.orig.rattrs, "orig", "exprs", "equations"}

    def evaluate(
        self, t: sym.Expression, *args: sym.MatrixSymbol
    ) -> tuple[sym.Expression, ...]:
        from orbitkit.symbolic.mappers import rename_variables

        result = (*self.exprs, *self.equations.values())
        return rename_variables(
            result,
            {vfrom: vto.name for vfrom, vto in zip(self.variables, args, strict=True)},
        )


def transform_distributed_delay_model(
    model: Model,
    n: int | tuple[int, ...] | None = None,
) -> ExtendedLinearChainTrickModel:
    """Transform the given *model* with distributed delays into one with only
    constant delays or a system of ODEs.

    See :func:`~orbitkit.models.linear_chain_tricks.transform_delay_kernels`.
    """
    from orbitkit.models.linear_chain_tricks import transform_delay_kernels

    if n is None:
        n = getattr(model, "n", None)

    if n is None:
        raise ValueError("must provide model size 'n'")

    _, exprs = model.symbolify(n)
    exprs, equations = transform_delay_kernels(exprs)

    return ExtendedLinearChainTrickModel(orig=model, exprs=exprs, equations=equations)


# }}}
