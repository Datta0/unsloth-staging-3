# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Behavioral guards for auto-load's local-model selection (#7374 / PR #7375).

Auto-load picks from ``GET /api/models/local`` with no user confirmation, and
that inventory also lists things llama.cpp cannot load as a main model: mmproj
projectors, MTP drafters, tail shards of a split GGUF, and partial downloads.
A bad pick burns a load attempt and is then remembered as the last used model.
These run the real helper through node, not a source grep.
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
    never see, so it must be retried before the smallest-cached-first sweep."""
    src = ADAPTER.read_text()
    auto_load = src.split("async function autoLoadSmallestModel", 1)[1]
    remembered = auto_load.index("tryAutoLoadRememberedLocalModel(localModels")
    cascade = auto_load.index("// GGUF first: smallest-total-size repo")
    sweep = auto_load.index("tryAutoLoadLocalModels(localModels)")
    assert remembered < cascade < sweep


def test_auto_load_follows_its_documented_tier_order():
    """``autoLoadSmallestModel``'s docstring promises: last-used, HF-cache GGUF,
    custom-folder / LM Studio / models_dir locals, then cached safetensors. A
    registered scan folder is a deliberate choice while an HF cache entry is a
    byproduct, so the local sweep runs ahead of the safetensors fallback."""
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
    /api/inference/validate response, plus the two outcome flags it owns."""
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
    """A custom scan folder can hold a LoRA adapter, and loading one makes the
    worker resolve ``base_model_name_or_path`` and pull the base from the Hub:
    the unsolicited download this path exists to remove. The refusal is a skip,
    not a failure, so both outcome flags must stay untouched (a later
    trust-blocked candidate still raises the consent dialog)."""
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


def test_companion_filtering_is_scoped_to_gguf_entries():
    """``isGgufCompanionPath`` reads GGUF filenames, so it may only judge GGUF
    rows: the backend predicates it mirrors (``hub/utils/gguf.py``
    ``is_mtp_drafter_path``, ``core/inference/llama_cpp.py``
    ``_is_companion_gguf_path``) return False for any non-``.gguf`` path, so a
    safetensors checkpoint in an ``mtp-...``/``MTP``/``mmproj`` folder is a real
    model. Every actual companion still carries a GGUF signal."""
    out = _run(
        "const nonGguf = [\n"
        "  { path: '/models/mtp-qwen3-next-80b-a3b', model_format: 'safetensors' },\n"
        "  { path: '/models/mmproj-training-run', model_format: 'safetensors' },\n"
        "  { path: '/models/MTP', model_format: 'safetensors' },\n"
        "  { path: '/models/DeepSeek-V4/mtp/stage2', model_format: 'safetensors' },\n"
        "];\n"
        "const companions = [\n"
        "  { path: '/m/a-GGUF/mmproj-F16.gguf', model_format: 'gguf' },\n"
        "  { path: '/m/mtp-qwen3-next-Q8_0.gguf', model_format: 'gguf' },\n"
        "  { path: '/m/Gemma-4-26B-A4B-GGUF/MTP', model_format: 'gguf' },\n"
        "  { path: '/m/gemma-Q4_K_M-00002-of-00002.gguf', model_format: 'gguf' },\n"
        "  { path: 'D:\\\\AI Models\\\\Gemma-GGUF\\\\mmproj-F32.gguf', model_format: 'gguf' },\n"
        "];\n"
        "console.log(JSON.stringify({\n"
        "  nonGguf: nonGguf.map((o) => helpers.isAutoLoadLocalModel(M(o))),\n"
        "  companions: companions.map((o) => helpers.isAutoLoadLocalModel(M(o))),\n"
        "  partialStillSkipped: helpers.isAutoLoadLocalModel(\n"
        "    M({ path: '/models/mtp-qwen3-next-80b-a3b', model_format: 'safetensors', partial: true })),\n"
        "}));\n"
    )
    assert out == {
        "nonGguf": [True] * 4,
        "companions": [False] * 5,
        "partialStillSkipped": False,
    }


def _run_local_gguf_variants(*, failing_quant: str, preferred: str | None = None):
    """Run the REAL ``tryAutoLoadLocalGgufModel`` over one folder with two
    downloaded quants where *failing_quant* rejects. Reports the attempts, the
    load result, the skip keys recorded, and what a second visit retries."""
    src = ADAPTER.read_text()
    start = src.index("  async function tryAutoLoadLocalGgufModel(")
    end = src.index("  async function tryAutoLoadRememberedLocalModel(", start)
    key_start = src.index("function autoLoadCandidateKey(")
    key_end = src.index("\n}\n", key_start) + 3
    return _run(
        "type LastLocalModelKind = 'gguf' | 'model';\n"
        "type LocalModelInfo = any;\n"
        f"{src[key_start:key_end]}\n"
        "const isDirectGgufPath = helpers.isDirectGgufPath;\n"
        "let hadNonTrustFailure = false;\n"
        "const skippedAutoLoadCandidates = new Set<string>();\n"
        "const attempted: string[] = [];\n"
        "function hasBigEndianGgufMarker(_p: string) { return false; }\n"
        "function isAutoLoadableGgufVariant(_v: any) { return true; }\n"
        "async function listGgufVariants(_id: string) {\n"
        "  return { variants: [\n"
        "    { quant: 'Q4_K_M', downloaded: true, size_bytes: 4000 },\n"
        "    { quant: 'Q2_K', downloaded: true, size_bytes: 1000 },\n"
        "  ] };\n"
        "}\n"
        f"const FAILING = {json.dumps(failing_quant)};\n"
        "async function loadAutoLoadCandidate(candidate: any): Promise<boolean> {\n"
        "  attempted.push(String(candidate.ggufVariant));\n"
        "  if (candidate.ggufVariant === FAILING) {\n"
        "    throw new Error('llama_model_load: error loading model');\n"
        "  }\n"
        "  return true;\n"
        "}\n"
        f"{src[start:end]}\n"
        "const model = M({ path: '/custom/MyRepo-GGUF', model_format: 'gguf' });\n"
        f"const preferred = {json.dumps(preferred)};\n"
        "const loaded = await tryAutoLoadLocalGgufModel(model, preferred);\n"
        "const firstPass = [...attempted];\n"
        "const skippedKeys = [...skippedAutoLoadCandidates];\n"
        "attempted.length = 0;\n"
        "await tryAutoLoadLocalGgufModel(model);\n"
        "console.log(JSON.stringify({\n"
        "  loaded, firstPass, skippedKeys, secondPass: [...attempted], hadNonTrustFailure,\n"
        "}));\n"
    )


def test_a_failed_local_gguf_variant_does_not_abort_the_rest_of_the_folder():
    """One corrupt quant must not strand the loadable ones beside it: the queue
    is smallest-first, so a bad Q2_K would otherwise unwind past the whole
    ``variantQueue`` loop even though the Q4_K_M in the same folder loads."""
    out = _run_local_gguf_variants(failing_quant = "Q2_K")
    assert out["loaded"] is True
    assert out["firstPass"] == ["Q2_K", "Q4_K_M"]
    assert out["hadNonTrustFailure"] is True


def test_a_failed_local_gguf_variant_is_remembered_by_its_own_quant():
    """The skip key must name the quant that actually failed: with no preferred
    variant the old key was ``gguf:<id>:``, which the per-variant lookup never
    matches, so a later sweep retried the same broken quant."""
    out = _run_local_gguf_variants(failing_quant = "Q2_K")
    assert out["skippedKeys"] == ["gguf:/custom/myrepo-gguf:q2_k"]
    assert "Q2_K" not in out["secondPass"]


def test_a_preferred_local_gguf_variant_still_leads_and_falls_back():
    """The remembered-variant ordering is unchanged, and a failure of the
    preferred quant still falls through to the remaining downloaded ones."""
    out = _run_local_gguf_variants(failing_quant = "Q4_K_M", preferred = "Q4_K_M")
    assert out["firstPass"] == ["Q4_K_M", "Q2_K"]
    assert out["loaded"] is True
    assert out["skippedKeys"] == ["gguf:/custom/myrepo-gguf:q4_k_m"]
