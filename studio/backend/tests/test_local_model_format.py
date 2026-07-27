# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for local GGUF ``model_format`` classification (PR #6364 follow-up).

Suffixless GGUF folders (custom folders / LM Studio) carry no ``-GGUF`` name
hint, so the scanners must surface ``model_format = "gguf"`` for the UI to route
them through the GGUF load path. The rule, shared by ``_dir_model_format`` and
``_scan_models_dir``: a directory is GGUF-format when it holds ``.gguf`` files
and no non-GGUF weights (``.safetensors`` / ``.bin``); a stray ``config.json``
must not disqualify it.

No GPU/network: only file names and sizes are inspected.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

# Keep runnable without optional logging deps (mirrors the sibling tests).
if "structlog" not in sys.modules:

    class _DummyLogger:
        def __getattr__(self, _name):
            return lambda *args, **kwargs: None

    sys.modules["structlog"] = types.SimpleNamespace(
        BoundLogger = _DummyLogger,
        get_logger = lambda *args, **kwargs: _DummyLogger(),
    )

import routes.models as models_route


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_bytes(b"\0")
    return path


def test_dir_model_format_gguf_only(tmp_path):
    d = tmp_path / "model"
    _touch(d / "model-Q4_K_M.gguf")
    assert models_route._dir_model_format(d) == "gguf"


def test_dir_model_format_mmproj_only_is_not_gguf(tmp_path):
    # A lone vision adapter has nothing servable: the variant selector drops mmproj.
    d = tmp_path / "model"
    _touch(d / "mmproj-F16.gguf")
    assert models_route._dir_model_format(d) is None


def test_dir_model_format_mmproj_beside_weights_is_still_gguf(tmp_path):
    d = tmp_path / "model"
    _touch(d / "mmproj-F16.gguf")
    _touch(d / "model-Q4_K_M.gguf")
    assert models_route._dir_model_format(d) == "gguf"


def test_dir_model_format_recursive_sees_split_quant_subdirs(tmp_path):
    # HF cache snapshots keep split quants in per-quant subdirs. A flat glob reports
    # no GGUF there, which would hide every sharded repo from the GGUF pickers.
    d = tmp_path / "snapshot"
    _touch(d / "UD-Q4_K_XL" / "model-00001-of-00002.gguf")
    assert models_route._dir_model_format(d) is None
    assert models_route._dir_model_format(d, recursive = True) == "gguf"


def test_dir_model_format_recursive_ignores_mmproj_only_subdirs(tmp_path):
    d = tmp_path / "snapshot"
    _touch(d / "mmproj" / "mmproj-F16.gguf")
    assert models_route._dir_model_format(d, recursive = True) is None


def test_scan_models_dir_mmproj_only_folder_is_not_gguf(tmp_path):
    # Same rule as _dir_model_format, applied by the parallel ./models scanner.
    _touch(tmp_path / "vision" / "mmproj-F16.gguf")
    _touch(tmp_path / "real" / "model-Q4_K_M.gguf")
    formats = {m.display_name: m.model_format for m in models_route._scan_models_dir(tmp_path)}
    assert formats["vision"] is None
    assert formats["real"] == "gguf"


def test_scan_models_dir_skips_standalone_mmproj_file(tmp_path):
    # A loose mmproj-*.gguf is a vision adapter with no weights to serve, so it must
    # not be offered as a model the way a loose primary GGUF is.
    _touch(tmp_path / "mmproj-F16.gguf")
    _touch(tmp_path / "model-Q4_K_M.gguf")
    names = {m.display_name for m in models_route._scan_models_dir(tmp_path)}
    assert names == {"model-Q4_K_M"}


def test_scan_lmstudio_dir_skips_standalone_mmproj_file(tmp_path):
    _touch(tmp_path / "mmproj-F16.gguf")
    _touch(tmp_path / "model-Q4_K_M.gguf")
    names = {m.display_name for m in models_route._scan_lmstudio_dir(tmp_path)}
    assert names == {"model-Q4_K_M"}


def test_scan_lmstudio_dir_skips_mmproj_under_publisher(tmp_path):
    # LM Studio's publisher/model.gguf layout classifies on a separate branch.
    _touch(tmp_path / "Publisher" / "mmproj-F16.gguf")
    _touch(tmp_path / "Publisher" / "model-Q4_K_M.gguf")
    names = {m.display_name for m in models_route._scan_lmstudio_dir(tmp_path)}
    assert names == {"model-Q4_K_M"}


def test_dir_model_format_gguf_with_config_is_still_gguf(tmp_path):
    # A config.json alongside the .gguf must not flip it to non-GGUF.
    d = tmp_path / "model"
    _touch(d / "config.json")
    _touch(d / "model-Q4_K_M.gguf")
    assert models_route._dir_model_format(d) == "gguf"


def test_dir_model_format_mixed_weights_is_mixed(tmp_path):
    # Real safetensors weights beside a GGUF: not a GGUF folder for the pickers,
    # which compare against "gguf", but not a plain checkpoint either, since a
    # load runs detect_gguf_model first and resolves it to the largest GGUF.
    d = tmp_path / "model"
    _touch(d / "model.safetensors")
    _touch(d / "model-Q4_K_M.gguf")
    assert models_route._dir_model_format(d) == "mixed"


def test_dir_model_format_no_gguf(tmp_path):
    d = tmp_path / "model"
    _touch(d / "config.json")
    _touch(d / "model.safetensors")
    assert models_route._dir_model_format(d) is None


def test_dir_model_format_ignores_tokenizer_bin(tmp_path):
    # A companion tokenizer.bin is not a weight file, so a GGUF folder shipping
    # one is still GGUF (not misread as a plain .bin checkpoint).
    d = tmp_path / "model"
    _touch(d / "tokenizer.bin")
    _touch(d / "model-Q4_K_M.gguf")
    assert models_route._dir_model_format(d) == "gguf"


def test_dir_model_format_weight_bin_is_mixed(tmp_path):
    # A real PyTorch weight .bin alongside a .gguf is the same mixed case.
    d = tmp_path / "model"
    _touch(d / "pytorch_model.bin")
    _touch(d / "model-Q4_K_M.gguf")
    assert models_route._dir_model_format(d) == "mixed"


def test_scan_models_dir_classifies_gguf_with_config(tmp_path):
    root = tmp_path / "models"
    # GGUF repo that also ships a config.json (the regression case).
    _touch(root / "gguf_repo" / "config.json")
    _touch(root / "gguf_repo" / "model-Q4_K_M.gguf")
    # A plain safetensors checkpoint stays non-GGUF.
    _touch(root / "st_repo" / "config.json")
    _touch(root / "st_repo" / "model.safetensors")
    # A standalone .gguf file is GGUF.
    _touch(root / "loose.gguf")

    fmt = {Path(m.path).name: m.model_format for m in models_route._scan_models_dir(root)}

    assert fmt["gguf_repo"] == "gguf"
    assert fmt["st_repo"] is None
    assert fmt["loose.gguf"] == "gguf"


def test_scan_models_dir_classifies_root_gguf_with_config(tmp_path):
    # Custom scan folders can point directly at a GGUF repo, not only at a
    # parent directory that contains model repos.
    root = tmp_path / "SuffixlessRepo"
    _touch(root / "config.json")
    _touch(root / "model-Q4_K_M.gguf")

    [row] = models_route._scan_models_dir(root)

    assert row.path == str(root)
    assert row.model_format == "gguf"


def test_scan_models_dir_reports_mixed_for_a_mixed_child(tmp_path):
    """The inline classification in _scan_models_dir must agree with
    _dir_model_format, since both feed the same model_format field."""
    child = tmp_path / "my-finetune"
    _touch(child / "config.json")
    _touch(child / "model.safetensors")
    _touch(child / "my-finetune-Q4_K_M.gguf")
    gguf_only = tmp_path / "my-finetune-q4"
    _touch(gguf_only / "my-finetune-Q4_K_M.gguf")
    plain = tmp_path / "my-finetune-full"
    _touch(plain / "config.json")
    _touch(plain / "model.safetensors")

    rows = {row.display_name: row.model_format for row in models_route._scan_models_dir(tmp_path)}

    assert rows["my-finetune"] == "mixed"
    assert rows["my-finetune-q4"] == "gguf"
    assert rows["my-finetune-full"] is None


def test_local_variants_drop_an_incomplete_split(tmp_path):
    """An interrupted split download must not be offered as a usable variant:
    its short byte total makes it look like the cheapest quant to try, while
    llama.cpp is handed shard 1 and fails on the missing siblings."""
    from utils.models.model_config import list_local_gguf_variants

    d = tmp_path / "repo"
    d.mkdir()
    # Complete two-shard Q4_K_M.
    _touch(d / "m-Q4_K_M-00001-of-00002.gguf")
    _touch(d / "m-Q4_K_M-00002-of-00002.gguf")
    # Interrupted three-shard Q8_0: only the first arrived.
    _touch(d / "m-Q8_0-00001-of-00003.gguf")
    # An unsplit quant is unaffected.
    _touch(d / "m-Q2_K.gguf")

    variants, _ = list_local_gguf_variants(str(d))

    assert {v.quant for v in variants} == {"Q4_K_M", "Q2_K"}


def test_local_variants_keep_a_complete_split(tmp_path):
    from utils.models.model_config import list_local_gguf_variants

    d = tmp_path / "repo"
    d.mkdir()
    for index in (1, 2, 3):
        _touch(d / f"m-Q8_0-{index:05d}-of-00003.gguf")

    variants, _ = list_local_gguf_variants(str(d))

    assert [v.quant for v in variants] == ["Q8_0"]


def test_weightless_local_rows_are_flagged_partial(tmp_path):
    """A folder admitted on its config alone has nothing to serve, so an
    unattended load would resolve the config, spend an attempt and then fail.
    It stays listed, flagged the way an incomplete cache download is, which the
    auto-load predicate already refuses."""
    weightless = tmp_path / "interrupted-copy"
    _touch(weightless / "config.json")
    real = tmp_path / "real-model"
    _touch(real / "config.json")
    _touch(real / "model.safetensors")
    gguf_only = tmp_path / "gguf-only"
    _touch(gguf_only / "m-Q4_K_M.gguf")

    rows = {r.display_name: r.partial for r in models_route._scan_models_dir(tmp_path)}

    assert rows["interrupted-copy"] is True
    assert rows["real-model"] is False
    assert rows["gguf-only"] is False


def test_weightless_lmstudio_rows_are_flagged_partial(tmp_path):
    """The LM Studio publisher branch admits on config alone in the same way."""
    _touch(tmp_path / "publisher" / "interrupted" / "config.json")
    _touch(tmp_path / "publisher" / "real" / "config.json")
    _touch(tmp_path / "publisher" / "real" / "model.safetensors")

    rows = {r.display_name: r.partial for r in models_route._scan_lmstudio_dir(tmp_path)}

    assert rows["interrupted"] is True
    assert rows["real"] is False


def test_projector_only_directory_is_flagged_partial(tmp_path):
    """A projector is not a servable weight, so a folder holding a config and an
    mmproj alone has nothing to load."""
    d = tmp_path / "vision-model"
    _touch(d / "config.json")
    _touch(d / "mmproj-F16.gguf")

    [row] = [r for r in models_route._scan_models_dir(tmp_path) if r.display_name == "vision-model"]

    assert row.partial is True


def test_lmstudio_bin_checkpoint_is_not_flagged_partial(tmp_path):
    """A PyTorch .bin checkpoint is real weights: _is_model_directory and
    _has_non_gguf_weights both recognise the name, so the LM Studio branch has
    to as well or a valid model is excluded from auto-load."""
    real = tmp_path / "publisher" / "bin-model"
    _touch(real / "config.json")
    _touch(real / "pytorch_model.bin")
    companion = tmp_path / "publisher" / "tokenizer-only"
    _touch(companion / "config.json")
    _touch(companion / "tokenizer.bin")

    rows = {r.display_name: r.partial for r in models_route._scan_lmstudio_dir(tmp_path)}

    assert rows["bin-model"] is False
    assert rows["tokenizer-only"] is True


def test_local_variants_drop_a_split_missing_its_first_shard(tmp_path):
    """Three distinct indices satisfy a bare count while shard 1 is absent."""
    from utils.models.model_config import list_local_gguf_variants

    d = tmp_path / "repo"
    d.mkdir()
    for index in (2, 3, 4):
        _touch(d / f"m-Q8_0-{index:05d}-of-00004.gguf")

    variants, _ = list_local_gguf_variants(str(d))

    assert variants == []


def test_local_variants_drop_mixed_split_generations(tmp_path):
    """An interrupted replacement leaves shards of two different splits under one
    quant label; neither set is whole, so the quant is not offered."""
    from utils.models.model_config import list_local_gguf_variants

    d = tmp_path / "repo"
    d.mkdir()
    _touch(d / "m-Q4_K_M-00001-of-00002.gguf")
    _touch(d / "m-Q4_K_M-00002-of-00003.gguf")
    _touch(d / "m-Q4_K_M-00003-of-00003.gguf")

    variants, _ = list_local_gguf_variants(str(d))

    assert variants == []
