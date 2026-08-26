# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""`UNSLOTH_FORCE_CUSTOM_DTYPE` is parsed by two separately released packages.

`unsloth/models/vision.py` and `unsloth_zoo/compiler.py` both do
`value.split(";", 4)` and both assert at least four separators. They ship on their own
schedules, so a user can run any new/old combination, and changing the layout on one
side would break the other on skew. The hardening deliberately kept the five fields
exactly as they were - it changed how the fields are *interpreted*, not what they are.

This pins that: the field count, the position of each field, and the fact that the
readers still agree. It also pins the direction of the trust decision, which is what
actually changed.

CPU-only and network-free.
"""

from __future__ import annotations

import ast
import pathlib
import re

import pytest

from unsloth.models._custom_dtype import register_custom_dtype, trusted_custom_dtype


LOADER = pathlib.Path(__import__("unsloth.models.loader", fromlist = ["x"]).__file__).read_text()


def _shipped_values() -> list[str]:
    tree = ast.parse(LOADER)
    values = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "register_custom_dtype":
                values.append(ast.literal_eval(node.args[0]))
    return values


@pytest.mark.parametrize("value", _shipped_values())
def test_every_shipped_value_has_five_fields(value):
    """Both readers do split(";", 4) after asserting count(";") >= 4."""
    assert value.count(";") >= 4
    fields = value.split(";", 4)
    assert len(fields) == 5


@pytest.mark.parametrize("value", _shipped_values())
def test_field_positions_are_unchanged(value):
    """checker, dtype, bnb_compute_dtype, custom_datatype, execute_code."""
    checker, dtype, bnb_compute_dtype, _custom, _execute = value.split(";", 4)
    assert checker in ("all", "float16", "torch.float16"), checker
    for field in (dtype, bnb_compute_dtype):
        assert re.fullmatch(r"(None|torch\.\w+)", field.strip()), field


def test_zoo_reader_still_parses_what_unsloth_writes():
    """The other package's parse of our values, reproduced exactly.

    unsloth_zoo is imported rather than copied so a layout change there fails here.
    """
    import unsloth_zoo.compiler as zoo_compiler

    source = pathlib.Path(zoo_compiler.__file__).read_text()
    assert (
        'custom_datatype.count(";") >= 4' in source
    ), "unsloth_zoo changed how it validates UNSLOTH_FORCE_CUSTOM_DTYPE"
    assert (
        'custom_datatype.split(";", 4)' in source
    ), "unsloth_zoo changed the field split for UNSLOTH_FORCE_CUSTOM_DTYPE"

    from unsloth_zoo.utils import _get_dtype

    for value in _shipped_values():
        _, dtype, _, _, _ = value.split(";", 4)
        resolved = _get_dtype(dtype.strip().removeprefix("torch."))
        expected = None if dtype.strip() == "None" else resolved
        assert resolved == expected


def test_trust_decision_is_on_the_value_we_set(monkeypatch):
    """The behaviour change, stated once: identical layout, different provenance.

    The two halves use different strings on purpose. `register_custom_dtype` records
    into a module-level set that is never cleared - a model can be loaded more than
    once - so registering a value here would make it trusted for the rest of the
    session, and reusing a value another test registered would make this one pass for
    the wrong reason.
    """
    template = _shipped_values()[0]
    checker, dtype, bnb, custom, execute = template.split(";", 4)

    inherited = ";".join([checker, dtype, bnb, custom, "import os; os.system('x')"])
    monkeypatch.setenv("UNSLOTH_FORCE_CUSTOM_DTYPE", inherited)
    value, trusted = trusted_custom_dtype()
    assert value == inherited  # dtype fields still readable
    assert not trusted  # code fields are not

    ours = ";".join([checker, dtype, bnb, custom, "pass  # only this test sets this"])
    register_custom_dtype(ours)
    value, trusted = trusted_custom_dtype()
    assert value == ours
    assert trusted


def test_no_producer_bypasses_the_registry():
    """A direct os.environ write would set the variable without registering it, so
    its own code fields would then be dropped as untrusted."""
    assert 'os.environ["UNSLOTH_FORCE_CUSTOM_DTYPE"]' not in LOADER
    assert "os.environ['UNSLOTH_FORCE_CUSTOM_DTYPE']" not in LOADER
