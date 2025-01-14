# SPDX-FileCopyrightText: 2025 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

from chaotique.typing import Array
from chaotique.utils import module_logger

log = module_logger(__name__)


# {{{ is_fixed_point


def is_fixed_point(x: Array, *, nlast: int | None = None) -> bool:
    if nlast is None:
        nlast = int(0.25 * x.shape[-1])

    return True

# }}}


# {{{ is_periodic


def is_periodic(x: Array, *, nlast: int | None = None) -> bool:
    if nlast is None:
        nlast = int(0.25 * x.shape[-1])

    return True

# }}}
