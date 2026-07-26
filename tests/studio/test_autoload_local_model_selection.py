# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Behavioral guards for auto-load's local-model selection (#7374 / PR #7375).

Auto-load now picks a model out of ``GET /api/models/local`` with no user
confirmation, and that inventory also lists things that are not loadable as a
main model: mmproj vision projectors, MTP drafters, tail shards of a split
GGUF, and half-downloaded copies. llama.cpp rejects every one of them
("missing tensor 'token_embd.weight'", "model must be loaded with the first
split"), so a bad pick burns a load attempt and is then remembered as the last
used model. These run the real helper through node, not a source grep.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

WORKDIR = Path(__file__).resolve().parents[2]


def _source_path(relative_path: str) -> Path:
    direct = WORKDIR / relative_path
    if direct.exists():
        return direct
    return WORKDIR / "unsloth_repo" / relative_path


HELPERS = _source_path("studio/frontend/src/features/chat/utils/auto-load-local-models.ts")
ADAPTER = _source_path("studio/frontend/src/features/chat/api/chat-adapter.ts")
TEMP = WORKDIR / "temp" / "autoload_local_model_selection"


def _require_node():
    if shutil.which("node") is None:
        pytest.skip("node not available")
    if not HELPERS.exists():
        pytest.skip("studio chat sources not present")
    result = subprocess.run(
        ["node", "--experimental-strip-types", "--version"],
        capture_output = True,
        text = True,
        timeout = 5,
    )
    if result.returncode != 0:
        pytest.skip("node --experimental-strip-types not available")


def _run(body: str):
    """Run *body* against the real helper module and parse its last stdout line."""
    _require_node()
    TEMP.mkdir(parents = True, exist_ok = True)
    module = os.path.relpath(HELPERS, TEMP).replace("\\", "/")
    script = TEMP / "run.mts"
    script.write_text(
        f'import * as helpers from "{module}";\n'
        "const M = (o) => ({\n"
        "  id: o.id ?? o.path,\n"
        "  display_name: o.display_name ?? 'model',\n"
        "  path: o.path,\n"
        "  source: o.source ?? 'custom',\n"
        "  ...o,\n"
        "});\n" + body
    )
    result = subprocess.run(
        ["node", "--experimental-strip-types", "--no-warnings", "run.mts"],
        cwd = str(TEMP),
        capture_output = True,
        text = True,
        timeout = 30,
        env = dict(os.environ, NODE_NO_WARNINGS = "1"),
    )
    assert result.returncode == 0, f"stderr: {result.stderr}\nstdout: {result.stdout}"
    return json.loads(result.stdout.strip().splitlines()[-1])


# Companion / non-primary GGUFs that must never be auto-selected.
REJECTED_PATHS = [
    "/mnt/ssd/Gemma-4-26B-A4B-GGUF/mmproj-F32.gguf",
    "/mnt/ssd/Gemma-4-26B-A4B-GGUF/mmproj-model-f16.gguf",
    "/mnt/ssd/Gemma-4-26B-A4B-GGUF/MTP",
    "/mnt/ssd/Gemma-4-26B-A4B-GGUF/MTP/gemma-4-26b-Q8_0-MTP.gguf",
    "/mnt/ssd/mtp-qwen3-next-Q8_0.gguf",
    "/mnt/ssd/gemma-Q4_K_M-00002-of-00002.gguf",
    "/mnt/ssd/gemma-Q4_K_M-011-of-011.gguf",
    # Windows separators and a UNC share reach the same rules.
    "D:\\AI Models\\Gemma-GGUF\\mmproj-F32.gguf",
    "\\\\nas\\share\\Gemma-GGUF\\MTP\\gemma-Q8_0-MTP.gguf",
]

# Real main models whose names merely look companion-ish.
ACCEPTED_PATHS = [
    "/mnt/ssd/Gemma-4-26B-A4B-GGUF",
    "/mnt/ssd/qwen3-4b-Q4_K_M.gguf",
    "/mnt/ssd/gemma-Q4_K_M-00001-of-00002.gguf",
    # "MTP" inside a repo name is a model family, not a drafter file.
    "/mnt/ssd/Qwen3.5-4B-MTP-GGUF",
    "/mnt/ssd/Qwen3.5-4B-MTP-GGUF/Qwen3.5-4B-UD-Q4_K_XL.gguf",
    "/mnt/ssd/Gemma-4-26B-A4B-GGUF/",
]


def test_companion_ggufs_are_never_auto_load_candidates():
    out = _run(
        f"const rejected = {json.dumps(REJECTED_PATHS)};\n"
        f"const accepted = {json.dumps(ACCEPTED_PATHS)};\n"
        "console.log(JSON.stringify({\n"
        "  rejected: rejected.map(helpers.isGgufCompanionPath),\n"
        "  accepted: accepted.map(helpers.isGgufCompanionPath),\n"
        "}));\n"
    )
    assert out["rejected"] == [True] * len(REJECTED_PATHS)
    assert out["accepted"] == [False] * len(ACCEPTED_PATHS)


def test_auto_load_skips_companions_partials_and_hf_cache():
    out = _run(
        "console.log(JSON.stringify({\n"
        "  mmproj: helpers.isAutoLoadLocalModel(M({ path: '/m/a-GGUF/mmproj-F16.gguf' })),\n"
        "  partial: helpers.isAutoLoadLocalModel(M({ path: '/m/a-GGUF', partial: true })),\n"
        "  hfCache: helpers.isAutoLoadLocalModel(M({ path: '/m/a-GGUF', source: 'hf_cache' })),\n"
        "  custom: helpers.isAutoLoadLocalModel(M({ path: '/m/a-GGUF' })),\n"
        "  lmstudio: helpers.isAutoLoadLocalModel(M({ path: '/m/a-GGUF', source: 'lmstudio' })),\n"
        "  modelsDir: helpers.isAutoLoadLocalModel(M({ path: '/m/a-GGUF', source: 'models_dir' })),\n"
        "}));\n"
    )
    assert out == {
        "mmproj": False,
        "partial": False,
        "hfCache": False,
        "custom": True,
        "lmstudio": True,
        "modelsDir": True,
    }


def test_gguf_classification_uses_the_entry_name_not_the_whole_path():
    """A parent folder like /mnt/my-GGUF-drive must not route a safetensors
    checkpoint through the GGUF variant path, which would drop it entirely."""
    out = _run(
        "console.log(JSON.stringify({\n"
        "  parentOnly: helpers.localModelIsGguf(M({\n"
        "    path: '/mnt/my-GGUF-drive/llama-3-8b', display_name: 'llama-3-8b' })),\n"
        "  ownName: helpers.localModelIsGguf(M({\n"
        "    path: '/mnt/ssd/Gemma-4-26B-A4B-GGUF', display_name: 'Gemma-4-26B-A4B-GGUF' })),\n"
        "  repoId: helpers.localModelIsGguf(M({\n"
        "    id: 'unsloth/gemma-3-1b-it-GGUF', path: '/hf/x', display_name: 'gemma-3-1b-it-GGUF' })),\n"
        "  formatHint: helpers.localModelIsGguf(M({\n"
        "    path: '/mnt/ssd/suffixless', display_name: 'suffixless', model_format: 'gguf' })),\n"
        "}));\n"
    )
    assert out == {"parentOnly": False, "ownName": True, "repoId": True, "formatHint": True}


def test_remembered_local_model_wins_over_the_cached_cascade():
    """The remembered model can be a custom-folder path the cached-repo lookups
    never see, so it must be retried against the local inventory before the
    smallest-cached-first sweep, not after it."""
    src = ADAPTER.read_text()
    auto_load = src.split("async function autoLoadSmallestModel", 1)[1]
    remembered = auto_load.index("tryAutoLoadRememberedLocalModel(localModels")
    cascade = auto_load.index("// GGUF first: smallest-total-size repo")
    sweep = auto_load.index("tryAutoLoadLocalModels(localModels)")
    assert remembered < cascade < sweep


def test_auto_load_follows_its_documented_tier_order():
    """``autoLoadSmallestModel``'s own docstring promises: last-used, then
    HF-cache GGUF, then custom-folder / LM Studio / models_dir locals, then
    cached safetensors. A registered scan folder is a deliberate "these are my
    models" choice while an HF cache entry is a byproduct of any past download,
    so the local sweep must run ahead of the safetensors fallback, not after it.
    """
    src = ADAPTER.read_text()
    assert "custom-folder / LM Studio / models-dir locals, then cached safetensors" in src
    auto_load = src.split("async function autoLoadSmallestModel", 1)[1]
    remembered = auto_load.index("tryAutoLoadRememberedLocalModel(localModels")
    cached_gguf = auto_load.index("// GGUF first: smallest-total-size repo")
    local_sweep = auto_load.index("tryAutoLoadLocalModels(localModels)")
    safetensors = auto_load.index("// Fall back to safetensors models.")
    assert remembered < cached_gguf < local_sweep < safetensors


def test_direct_gguf_autoload_keeps_the_big_endian_guard():
    """Standalone .gguf files bypass the variant list, so they need the same
    big-endian filename guard the cached-variant path applies."""
    src = ADAPTER.read_text()
    direct = src.split("if (isDirectGgufPath(model.path)) {", 1)[1][:600]
    assert "hasBigEndianGgufMarker(model.path)" in direct


def _run_can_auto_load(validation: dict):
    """Run the REAL canAutoLoad closure from chat-adapter.ts against a stubbed
    /api/inference/validate response, and report its answer plus the two
    outcome flags it owns."""
    src = ADAPTER.read_text()
    start = src.index("  async function canAutoLoad(")
    end = src.index("  async function loadAutoLoadCandidate(", start)
    return _run(
        "let blockedByTrustRemoteCode = false;\n"
        "let hadNonTrustFailure = false;\n"
        "const hfToken = null;\n"
        "const trustRemoteCode = false;\n"
        f"const VALIDATION = {json.dumps(validation)};\n"
        "async function validateModel(_payload) { return VALIDATION; }\n"
        f"{src[start:end]}\n"
        "const ok = await canAutoLoad({\n"
        "  model_path: '/mnt/ssd/my-finetune-lora', max_seq_length: 4096, is_lora: false });\n"
        "console.log(JSON.stringify({ ok, blockedByTrustRemoteCode, hadNonTrustFailure }));\n"
    )


def test_background_auto_load_refuses_lora_adapters():
    """A custom scan folder can hold a LoRA adapter: the inventory lists any
    ``adapter_config.json`` directory as a plain local row with no GGUF hint, so
    the source filter admits it. Loading one makes the worker resolve
    ``base_model_name_or_path`` and pull the base from the Hub, which is the
    unsolicited download this path exists to remove. validate already reports
    ``is_lora``, so the background gate must refuse before /load.

    The refusal is a skip, not a failure: it must leave both outcome flags
    alone so a later trust-blocked candidate still raises the consent dialog.
    """
    assert _run_can_auto_load({"is_lora": True}) == {
        "ok": False,
        "blockedByTrustRemoteCode": False,
        "hadNonTrustFailure": False,
    }


def test_background_auto_load_keeps_the_existing_validate_gates():
    """The LoRA skip must not disturb the consent/upgrade gates beside it."""
    assert _run_can_auto_load({}) == {
        "ok": True,
        "blockedByTrustRemoteCode": False,
        "hadNonTrustFailure": False,
    }
    assert _run_can_auto_load({"requires_trust_remote_code": True}) == {
        "ok": False,
        "blockedByTrustRemoteCode": True,
        "hadNonTrustFailure": False,
    }
    assert _run_can_auto_load({"requires_security_review": True}) == {
        "ok": False,
        "blockedByTrustRemoteCode": True,
        "hadNonTrustFailure": False,
    }
    assert _run_can_auto_load({"requires_transformers_upgrade": True}) == {
        "ok": False,
        "blockedByTrustRemoteCode": False,
        "hadNonTrustFailure": True,
    }
