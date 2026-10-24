# SPDX-FileCopyrightText: 2025 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging
from importlib import metadata

logging.getLogger(__name__).addHandler(logging.NullHandler())

__version__ = metadata.version("orbitkit")
