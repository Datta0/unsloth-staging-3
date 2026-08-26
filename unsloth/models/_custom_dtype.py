# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""`UNSLOTH_FORCE_CUSTOM_DTYPE` carries executable code in two of its five fields.

Every value unsloth ships is a literal in `models/loader.py`, and the only readers
(`models/vision.py`, `unsloth_zoo/compiler.py`) run in the same process that sets it.
So the variable is really a process-local global that happens to travel through
`os.environ`, and a value that was *not* set by this process has no legitimate source.

`register_custom_dtype` records what we set; `trusted_custom_dtype` hands the value
back only if it matches. An inherited or externally planted environment still selects
dtypes (harmless, and covered by the fixed table below), but its code fields are
dropped instead of executed.

The wire format is unchanged. `unsloth_zoo` parses the same five fields, and the two
packages are versioned separately, so changing the layout here would break on skew.
"""

__all__ = [
    "register_custom_dtype",
    "trusted_custom_dtype",
    "resolve_dtype",
    "DTYPE_ALIASES",
]

import os

import torch

# The dtype fields are a closed set: every `UNSLOTH_FORCE_CUSTOM_DTYPE` literal in
# loader.py uses `None`, `torch.float16` or `torch.bfloat16`. A table instead of
# `eval` means the field names a dtype rather than being an arbitrary expression.
DTYPE_ALIASES = {
    "None": None,
    "none": None,
    "": None,
    "torch.float16": torch.float16,
    "torch.bfloat16": torch.bfloat16,
    "torch.float32": torch.float32,
    "torch.float64": torch.float64,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float64": torch.float64,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}

_ENV_KEY = "UNSLOTH_FORCE_CUSTOM_DTYPE"

# Values this process set. Not cleared: a model can be loaded more than once, and the
# set only ever holds our own literals.
_REGISTERED = set()


def resolve_dtype(text):
    """Maps a dtype field of `UNSLOTH_FORCE_CUSTOM_DTYPE` onto a torch dtype"""
    key = str(text).strip()
    if key not in DTYPE_ALIASES:
        raise ValueError(
            f"Unsloth: `{_ENV_KEY}` names an unsupported dtype `{key}`.\n"
            f"Supported: {sorted(x for x in DTYPE_ALIASES if x)}"
        )
    return DTYPE_ALIASES[key]


def register_custom_dtype(value):
    """Sets `UNSLOTH_FORCE_CUSTOM_DTYPE` and records it as ours"""
    _REGISTERED.add(value)
    os.environ[_ENV_KEY] = value
    return value


def trusted_custom_dtype():
    """Returns (value, code_is_trusted).

    `code_is_trusted` is False for a value this process did not set, which is the
    signal to honour the dtype fields but drop the two code fields.
    """
    value = os.environ.get(_ENV_KEY, "")
    if value == "":
        return "", False
    return value, value in _REGISTERED
