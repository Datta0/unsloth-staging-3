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
RUNTIME = _source_path("studio/frontend/src/features/chat/hooks/use-chat-model-runtime.ts")
STORE = _source_path("studio/frontend/src/features/chat/stores/chat-runtime-store.ts")
EXTERNAL = _source_path("studio/frontend/src/features/chat/external-providers.ts")
TEMP = WORKDIR / "temp" / "autoload_local_model_selection"


def _slice(src: str, start_marker: str, end_marker: str) -> str:
    start = src.index(start_marker)
    return src[start : src.index(end_marker, start)]


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
    model. Every actual companion still carries a GGUF signal.

    The rows use the vocabulary ``/api/models/local`` actually reports: a folder
    holding non-GGUF weights comes back with no ``model_format`` at all, because
    ``_scan_models_dir`` and ``_dir_model_format`` emit ``"gguf"`` or nothing."""
    out = _run(
        "const nonGguf = [\n"
        "  { path: '/models/mtp-qwen3-next-80b-a3b' },\n"
        "  { path: '/models/mmproj-training-run' },\n"
        "  { path: '/models/MTP' },\n"
        "  { path: '/models/DeepSeek-V4/mtp/stage2' },\n"
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
        "    M({ path: '/models/mtp-qwen3-next-80b-a3b', partial: true })),\n"
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
        "const cachedRepoLoadDirs: any[] = [];\n"
        "const hostRunsMlx = false;\n"
        "const isUnsupportedMlxLocalModel = helpers.isUnsupportedMlxLocalModel;\n"
        # The local tiers consult the skip set through these helpers, so the
        # sliced functions need the real implementations, not stand-ins.
        f"{_slice(src, '  function isPathInside(', '  async function canAutoLoad(')}\n"
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


def _run_remembered_local(
    *,
    kind: str,
    gguf_variant: str | None = None,
    name: str = "MyModel",
    is_mac: bool = False,
):
    """Run the REAL ``tryAutoLoadRememberedLocalModel`` against a remembered entry
    whose load throws, then report the skip keys it recorded."""
    src = ADAPTER.read_text()
    start = src.index("  async function tryAutoLoadRememberedLocalModel(")
    end = src.index("  async function tryAutoLoadLocalModels(", start)
    key_start = src.index("function autoLoadCandidateKey(")
    key_end = src.index("\n}\n", key_start) + 3
    return _run(
        "type LastLocalModelKind = 'gguf' | 'model';\n"
        "type LocalModelInfo = any;\n"
        f"{src[key_start:key_end]}\n"
        "let hadNonTrustFailure = false;\n"
        "const skippedAutoLoadCandidates = new Set<string>();\n"
        "const cachedRepoLoadDirs: any[] = [];\n"
        f"const hostRunsMlx = {json.dumps(is_mac)};\n"
        "const isUnsupportedMlxLocalModel = helpers.isUnsupportedMlxLocalModel;\n"
        # The local tiers consult the skip set through these helpers, so the
        # sliced functions need the real implementations, not stand-ins.
        f"{_slice(src, '  function isPathInside(', '  async function canAutoLoad(')}\n"
        "const toastId = 't';\n"
        "const toast: any = () => {};\n"
        "const store = { params: { maxSeqLength: 4096 } };\n"
        "const findLocalModel = (models: any[], id: string) =>\n"
        "  models.find((m) => m.id === id) ?? null;\n"
        "const isAutoLoadLocalModel = () => true;\n"
        "const localModelIsGguf = (m: any) => m.model_format === 'gguf';\n"
        "async function tryAutoLoadLocalGgufModel(_m: any, _v?: any): Promise<boolean> {\n"
        "  throw new Error('llama_model_load: error loading model');\n"
        "}\n"
        "async function loadAutoLoadCandidate(_c: any): Promise<boolean> {\n"
        "  throw new Error('llama_model_load: error loading model');\n"
        "}\n"
        f"{src[start:end]}\n"
        f"const kind = {json.dumps(kind)};\n"
        f"const ggufVariant = {json.dumps(gguf_variant)};\n"
        f"const name = {json.dumps(name)};\n"
        "const models = [{\n"
        "  id: `/custom/${name}`, model_id: name, display_name: name,\n"
        "  path: `/custom/${name}`,\n"
        "  active_cache: false,\n"
        # Only "gguf" or nothing: the vocabulary /api/models/local reports.
        "  model_format: kind === 'gguf' ? 'gguf' : undefined,\n"
        "}];\n"
        "const loaded = await tryAutoLoadRememberedLocalModel(models, {\n"
        "  id: `/custom/${name}`, kind, ggufVariant,\n"
        "});\n"
        "console.log(JSON.stringify({\n"
        "  loaded, skippedKeys: [...skippedAutoLoadCandidates], hadNonTrustFailure,\n"
        "}));\n"
    )


@pytest.mark.parametrize(
    "kind,gguf_variant,expected_keys",
    [
        # A standalone GGUF: the sweep's own check keys on a null variant. No
        # repo-id alias here, because the row is not inside any cached
        # candidate's cache dir, so its model_id is only a name.
        ("gguf", None, ["gguf:/custom/mymodel:"]),
        ("gguf", "Q4_K_M", ["gguf:/custom/mymodel:q4_k_m"]),
        ("model", None, ["model:/custom/mymodel:"]),
    ],
)
def test_a_failed_remembered_local_model_is_not_retried_by_the_sweep(
    kind, gguf_variant, expected_keys
):
    """The sweep walks the same inventory, so a remembered entry that threw would
    consume a second slot of the three-attempt budget and could starve a valid
    model later in the list."""
    out = _run_remembered_local(kind = kind, gguf_variant = gguf_variant)

    assert out["loaded"] is False
    assert out["hadNonTrustFailure"] is True
    assert out["skippedKeys"] == expected_keys


def _run_local_sweep(
    models: list[dict],
    *,
    failing: list[str],
    gguf_folder_fail: str | None = None,
    preskipped: list[str] | None = None,
    is_mac: bool = False,
    cached_repo_dirs: list[dict] | None = None,
):
    """Run the REAL ``tryAutoLoadLocalModels`` (driving the real
    ``tryAutoLoadLocalGgufModel``) over *models*, failing every load whose id is
    in *failing*. Reports what was attempted, the budget spent, and the skip
    keys recorded."""
    src = ADAPTER.read_text()
    key = _slice(src, "function autoLoadCandidateKey(", "\n}\n") + "\n}\n"
    gguf_fn = _slice(
        src,
        "  async function tryAutoLoadLocalGgufModel(",
        "  async function tryAutoLoadRememberedLocalModel(",
    )
    sweep_fn = _slice(
        src,
        "  async function tryAutoLoadLocalModels(",
        "  try {\n    const [ggufRepos",
    )
    return _run(
        "type LastLocalModelKind = 'gguf' | 'model';\n"
        "type LocalModelInfo = any;\n"
        f"{key}\n"
        "const MAX_AUTO_LOAD_ATTEMPTS = 3;\n"
        "let loadAttempts = 0;\n"
        "let hadNonTrustFailure = false;\n"
        f"const skippedAutoLoadCandidates = new Set<string>({json.dumps(preskipped or [])});\n"
        # The local tiers consult the skip set through these helpers, so the
        # sliced functions need the real implementations, not stand-ins.
        f"{_slice(src, '  function isPathInside(', '  async function canAutoLoad(')}\n"
        "const attempted: string[] = [];\n"
        "const isDirectGgufPath = helpers.isDirectGgufPath;\n"
        "const localModelIsGguf = helpers.localModelIsGguf;\n"
        "const isAutoLoadLocalModel = helpers.isAutoLoadLocalModel;\n"
        "const isUnsupportedMlxLocalModel = helpers.isUnsupportedMlxLocalModel;\n"
        f"const hostRunsMlx = {json.dumps(is_mac)};\n"
        f"const cachedRepoLoadDirs = {json.dumps(cached_repo_dirs or [])};\n"
        "const sortLocalModelsForAutoLoad = helpers.sortLocalModelsForAutoLoad;\n"
        "function hasBigEndianGgufMarker(_p: string) { return false; }\n"
        "function isAutoLoadableGgufVariant(_v: any) { return true; }\n"
        f"const FOLDER_FAIL = {json.dumps(gguf_folder_fail)};\n"
        "async function listGgufVariants(_id: string) {\n"
        "  return { variants: [{ quant: 'Q4_K_M', downloaded: true, size_bytes: 4000 }] };\n"
        "}\n"
        f"const FAILING = new Set({json.dumps(failing)});\n"
        "async function loadAutoLoadCandidate(candidate: any): Promise<boolean> {\n"
        "  if (loadAttempts >= MAX_AUTO_LOAD_ATTEMPTS) { return false; }\n"
        "  attempted.push(\n"
        "    `${candidate.kind}|${candidate.id}|${candidate.ggufVariant ?? ''}`,\n"
        "  );\n"
        "  loadAttempts += 1;\n"
        "  if (FAILING.has(candidate.id) || candidate.id === FOLDER_FAIL) {\n"
        "    throw new Error('llama_model_load: error loading model');\n"
        "  }\n"
        "  return true;\n"
        "}\n"
        f"{gguf_fn}\n"
        f"{sweep_fn}\n"
        f"const models = {json.dumps(models)}.map(M);\n"
        "const loaded = await tryAutoLoadLocalModels(models);\n"
        "console.log(JSON.stringify({\n"
        "  loaded, attempted, loadAttempts, hadNonTrustFailure,\n"
        "  skippedKeys: [...skippedAutoLoadCandidates],\n"
        "}));\n"
    )


# ``collect_local_models`` keeps a custom-folder row beside the models-dir /
# LM Studio row for the same path (the dedup key is (id, source) for custom),
# so the sweep walks the identical entry twice.
DUPLICATE_SAFETENSORS = [
    {"path": "/scan/AModel", "source": "models_dir", "display_name": "AModel"},
    {"path": "/scan/AModel", "source": "custom", "display_name": "AModel"},
    {"path": "/scan/ZGood", "source": "custom", "display_name": "ZGood"},
]

DUPLICATE_DIRECT_GGUF = [
    {"path": "/scan/broken-Q4_K_M.gguf", "source": "lmstudio", "display_name": "broken"},
    {"path": "/scan/broken-Q4_K_M.gguf", "source": "custom", "display_name": "broken"},
    {"path": "/scan/good-Q4_K_M.gguf", "source": "custom", "display_name": "good"},
]


def test_a_failed_local_checkpoint_is_not_retried_by_its_duplicate_row():
    """The sweep's catch has to name the key the next visit checks. Without it
    the overlapping custom-folder row reloads the identical broken model and
    spends a second slot of the three-attempt budget."""
    out = _run_local_sweep(DUPLICATE_SAFETENSORS, failing = ["/scan/AModel"])

    assert out["attempted"] == ["model|/scan/AModel|", "model|/scan/ZGood|"]
    assert out["skippedKeys"] == ["model:/scan/amodel:"]
    assert out["loaded"] is True
    assert out["loadAttempts"] == 2
    assert out["hadNonTrustFailure"] is True


def test_a_failed_standalone_gguf_is_not_retried_by_its_duplicate_row():
    """The direct-``.gguf`` branch loads outside ``tryAutoLoadLocalGgufModel``'s
    own try, so its throw lands in the sweep's catch and must be recorded under
    the null-variant key that branch guards on."""
    out = _run_local_sweep(DUPLICATE_DIRECT_GGUF, failing = ["/scan/broken-Q4_K_M.gguf"])

    assert out["attempted"] == ["gguf|/scan/broken-Q4_K_M.gguf|", "gguf|/scan/good-Q4_K_M.gguf|"]
    assert out["skippedKeys"] == ["gguf:/scan/broken-q4_k_m.gguf:"]
    assert out["loaded"] is True
    assert out["loadAttempts"] == 2


def test_a_failing_gguf_folder_still_records_only_the_quant_that_failed():
    """The variant loop owns its own catches, so the sweep must not layer a
    folder-wide key on top of the per-quant one and blacklist good siblings."""
    out = _run_local_sweep(
        [
            {"path": "/scan/RepoA-GGUF", "source": "custom", "display_name": "RepoA-GGUF"},
            {"path": "/scan/RepoB-GGUF", "source": "custom", "display_name": "RepoB-GGUF"},
        ],
        failing = [],
        gguf_folder_fail = "/scan/RepoA-GGUF",
    )

    assert out["skippedKeys"] == ["gguf:/scan/repoa-gguf:q4_k_m"]
    assert out["loaded"] is True


def test_a_clean_sweep_blacklists_nothing():
    """Nothing failed, so nothing may be recorded as skipped."""
    out = _run_local_sweep(DUPLICATE_SAFETENSORS, failing = [])

    assert out["attempted"] == ["model|/scan/AModel|"]
    assert out["skippedKeys"] == []
    assert out["loaded"] is True
    assert out["hadNonTrustFailure"] is False


# Registering an HF cache as a scan folder surfaces each repo twice: the cached
# tiers address it by repo id, collect_local_models keeps a custom row for the
# same snapshot addressed by absolute path. Only cache-derived rows carry
# model_id, so it is an exact alias between the two spellings.
# What listCachedGguf reported: the snapshot path this copy loads from. A row
# without a load_id resolves by repo id through the active cache and never
# reaches this table.
CACHED_REPO_DIRS = [
    {"repoId": "Org/Foo-GGUF", "loadPath": "/hub/models--Org--Foo-GGUF"},
]

REGISTERED_CACHE_ROWS = [
    {
        "id": "/hub/models--Org--Foo-GGUF/snapshots/rev0",
        "path": "/hub/models--Org--Foo-GGUF/snapshots/rev0",
        "model_id": "Org/Foo-GGUF",
        "display_name": "Foo-GGUF",
        "source": "custom",
        # Set by _scan_hf_cache alone: this row is the cache duplicate.
        "active_cache": False,
        "model_format": "gguf",
        "updated_at": 2000,
    },
    {
        "path": "/scan/ZGood-GGUF",
        "source": "custom",
        "display_name": "ZGood-GGUF",
        "model_format": "gguf",
        "updated_at": 1000,
    },
]


def test_a_cached_failure_is_not_retried_by_its_registered_cache_row():
    """The cached GGUF tier runs first and records its failure under the repo
    id. The same snapshot arrives in the sweep under its absolute path, so
    without the alias it spends a second slot of the three-attempt budget."""
    out = _run_local_sweep(
        REGISTERED_CACHE_ROWS,
        failing = [],
        preskipped = ["gguf:org/foo-gguf:q4_k_m"],
        cached_repo_dirs = CACHED_REPO_DIRS,
    )

    assert out["attempted"] == ["gguf|/scan/ZGood-GGUF|Q4_K_M"]
    assert out["loadAttempts"] == 1
    assert out["loaded"] is True


def test_a_registered_cache_row_records_both_spellings_when_it_fails():
    """Recorded the other way too, so the cached safetensors fallback that runs
    after the sweep does not retry the same snapshot under its Hub id."""
    out = _run_local_sweep(
        REGISTERED_CACHE_ROWS,
        failing = [],
        gguf_folder_fail = "/hub/models--Org--Foo-GGUF/snapshots/rev0",
        cached_repo_dirs = CACHED_REPO_DIRS,
    )

    assert out["skippedKeys"] == [
        "gguf:/hub/models--org--foo-gguf/snapshots/rev0:q4_k_m",
        "gguf:org/foo-gguf:q4_k_m",
    ]
    assert out["loaded"] is True


# A scan folder that is itself an HF cache surfaces snapshot rows as "custom".
# Their id/path is the snapshot dir and their display_name is the bare repo
# name, so ``model_format`` is the only GGUF signal a suffixless repo has.
HF_CACHE_SNAPSHOT = {
    "path": "/scan/models--acme--tiny-model/snapshots/rev",
    "display_name": "tiny-model",
    "source": "custom",
}


def test_a_cached_snapshot_needs_the_backend_format_hint_to_load_as_gguf():
    """Without the hint the sweep sends a GGUF snapshot down the checkpoint
    path, where the backend picks the largest ``.gguf`` in the folder rather
    than the smallest downloaded variant, and the run is recorded as
    safetensors with a 4096-token limit."""
    unhinted = _run_local_sweep([HF_CACHE_SNAPSHOT], failing = [])
    hinted = _run_local_sweep([{**HF_CACHE_SNAPSHOT, "model_format": "gguf"}], failing = [])

    assert unhinted["attempted"] == ["model|/scan/models--acme--tiny-model/snapshots/rev|"]
    assert hinted["attempted"] == ["gguf|/scan/models--acme--tiny-model/snapshots/rev|Q4_K_M"]


def _run_manual_load_record_guard(cases: list[dict]):
    """Evaluate the REAL manual-load recorder guard from use-chat-model-runtime
    against the real ``isLocalModelPath`` / ``isExternalModelId`` predicates."""
    src = RUNTIME.read_text()
    start = src.index("            if (\n              !isLora &&")
    open_paren = src.index("(", start)
    condition = src[open_paren + 1 : src.index("\n            ) {", open_paren)]
    store_src = STORE.read_text()
    ext_src = EXTERNAL.read_text()
    return _run(
        f"{_slice(ext_src, 'const EXTERNAL_MODEL_PREFIX =', chr(10))}\n"
        f"{_slice(store_src, 'export function isLocalModelPath(', chr(10) + '}' + chr(10))}\n}}\n"
        f"{_slice(ext_src, 'export function isExternalModelId(', chr(10) + '}' + chr(10))}\n}}\n"
        "function shouldRecord(c: any): boolean {\n"
        "  const isLora = c.isLora ?? false;\n"
        "  const loadResponse = c.loadResponse ?? {};\n"
        "  const nativePathToken = c.nativePathToken ?? undefined;\n"
        "  const modelId = c.modelId;\n"
        f"  return Boolean({condition});\n"
        "}\n"
        f"const cases = {json.dumps(cases)};\n"
        "console.log(JSON.stringify(cases.map(shouldRecord)));\n"
    )


MANUAL_LOAD_CASES = [
    # Inventory-backed local picks: auto-load restores these from the local
    # sweep, so they have to be remembered like a cached repo id.
    ({"modelId": "/scan/MyRepo-GGUF"}, True),
    ({"modelId": "/scan/qwen3-4b-Q4_K_M.gguf"}, True),
    ({"modelId": "D:\\AI Models\\MyRepo-GGUF"}, True),
    ({"modelId": "/home/u/.lmstudio/models/org/MyModel"}, True),
    # A native file-picker pick is valid only for the lease that just expired.
    ({"modelId": "/tmp/dropped.gguf", "nativePathToken": "tok-123"}, False),
    # Adapters would make the backend pull their base model from the Hub.
    ({"modelId": "/scan/MyAdapter", "isLora": True}, False),
    ({"modelId": "/scan/MyAdapter2", "loadResponse": {"is_lora": True}}, False),
    # External providers are not local models at all.
    ({"modelId": "external::openai::gpt-4o-mini"}, False),
    # Unchanged: a cached hub repo is still remembered.
    ({"modelId": "unsloth/gemma-3-4b-it"}, True),
]


def test_manual_local_picks_are_remembered_as_the_last_used_model():
    """``tryAutoLoadRememberedLocalModel`` can only restore what the manual load
    path records. Gating on the path shape meant a custom-folder pick was never
    written, so the next start restored a stale cached repo instead."""
    out = _run_manual_load_record_guard([case for case, _ in MANUAL_LOAD_CASES])

    assert out == [expected for _, expected in MANUAL_LOAD_CASES]


def test_local_inventory_reports_only_its_own_format_vocabulary():
    """The companion scoping keys off a positive GGUF classification, which is
    only sound because ``/api/models/local`` has no non-GGUF vocabulary beyond
    the ambiguous ``"mixed"``. Pin what the route can emit: ``"gguf"``,
    ``"mixed"``, ``None``, or a ``_dir_model_format`` call returning those.

    The richer set ("safetensors", "adapter", "unknown") is the hub inventory
    schema, which feeds the picker rather than auto-load. Reintroducing an
    allowlist of those names here silently matches nothing."""
    source = _source_path("studio/backend/routes/models.py").read_text()
    assignments = {
        line.strip() for line in source.splitlines() if line.strip().startswith("model_format =")
    }
    assert assignments, "no model_format assignments found; path moved?"
    allowed = {
        'model_format = "gguf"',
        'model_format = "gguf",',
        "model_format = None",
        'model_format = "mixed" if has_non_gguf_weights else "gguf"',
        "model_format = model_format,",
        "model_format = (",
    }
    for assignment in assignments:
        assert assignment in allowed or assignment.startswith(
            "model_format = _dir_model_format("
        ), assignment
    for dead in ('"safetensors"', '"adapter"', '"checkpoint"', '"unknown"'):
        assert f"model_format = {dead}" not in source
    # _dir_model_format itself returns only these.
    body = _slice(source, "def _dir_model_format(", "\ndef ")
    returns = {line.strip() for line in body.splitlines() if line.strip().startswith("return")}
    assert returns == {
        "return None",
        'return "mixed" if _has_non_gguf_weights(path) else "gguf"',
    }, returns


def test_a_mixed_weight_directory_is_left_to_an_explicit_pick():
    """Loading such a folder resolves to a GGUF either way, because
    from_identifier runs detect_gguf_model on any local path first and it takes
    the largest file, so the checkpoint path would load the biggest quant and
    record the run as non-GGUF. Unattended auto-load skips it."""
    out = _run(
        "console.log(JSON.stringify({\n"
        "  mixed: helpers.isAutoLoadLocalModel(M({\n"
        "    path: '/models/my-finetune', model_format: 'mixed' })),\n"
        "  pureGguf: helpers.isAutoLoadLocalModel(M({\n"
        "    path: '/models/my-finetune-q4', model_format: 'gguf' })),\n"
        "  pureCheckpoint: helpers.isAutoLoadLocalModel(M({\n"
        "    path: '/models/my-finetune-full' })),\n"
        # A mixed row is not GGUF for classification either.
        "  mixedIsNotGguf: helpers.localModelIsGguf(M({\n"
        "    path: '/models/my-finetune', display_name: 'my-finetune',\n"
        "    model_format: 'mixed' })),\n"
        "}));\n"
    )
    assert out == {
        "mixed": False,
        "pureGguf": True,
        "pureCheckpoint": True,
        "mixedIsNotGguf": False,
    }


def test_a_companion_named_checkpoint_is_still_loadable():
    """The bug the scoping exists for: a real non-GGUF checkpoint whose folder
    is named like a companion. The scanner reports no format for it (it holds no
    GGUF), so the companion predicate must not judge it."""
    out = _run(
        "console.log(JSON.stringify({\n"
        # No format: a real checkpoint, admitted.
        "  mtpNamedCheckpoint: helpers.isAutoLoadLocalModel(M({\n"
        "    path: '/models/mtp-qwen3-next-80b-a3b' })),\n"
        "  mmprojNamedCheckpoint: helpers.isAutoLoadLocalModel(M({\n"
        "    path: '/models/mmproj-training-run' })),\n"
        # Classified gguf: a real companion, still refused. A folder whose only
        # GGUFs are MTP drafters is classified "gguf" (_is_main_gguf_filename
        # drops mmproj only), so this covers the drafter directory too.
        "  ggufMtpDir: helpers.isAutoLoadLocalModel(M({\n"
        "    path: '/models/Gemma-4-26B-A4B-GGUF/MTP', model_format: 'gguf' })),\n"
        "  ggufMmprojFile: helpers.isAutoLoadLocalModel(M({\n"
        "    path: '/models/repo/mmproj-F16.gguf', model_format: 'gguf' })),\n"
        # A .gguf path classifies itself, with or without the hint.
        "  bareMmprojFile: helpers.isAutoLoadLocalModel(M({\n"
        "    path: '/models/repo/mmproj-F16.gguf' })),\n"
        "  bareTailShard: helpers.isAutoLoadLocalModel(M({\n"
        "    path: '/models/repo/gemma-Q4_K_M-00002-of-00002.gguf' })),\n"
        # A -GGUF repo name classifies itself too.
        "  suffixNamedMtpDir: helpers.isAutoLoadLocalModel(M({\n"
        "    path: '/models/Gemma-GGUF/MTP', display_name: 'Gemma-GGUF' })),\n"
        "}));\n"
    )
    assert out == {
        "mtpNamedCheckpoint": True,
        "mmprojNamedCheckpoint": True,
        "ggufMtpDir": False,
        "ggufMmprojFile": False,
        "bareMmprojFile": False,
        "bareTailShard": False,
        "suffixNamedMtpDir": False,
    }


# An LM Studio row carries a `publisher/model-name` model_id of the same shape as
# a Hub repo id, for an independent copy with its own files on disk.
LMSTUDIO_LOOKALIKE = [
    {
        "id": "/lmstudio/unsloth/gemma-3-4b-it-GGUF",
        "path": "/lmstudio/unsloth/gemma-3-4b-it-GGUF",
        "model_id": "unsloth/gemma-3-4b-it-GGUF",
        "display_name": "gemma-3-4b-it-GGUF",
        "source": "lmstudio",
        "model_format": "gguf",
        "updated_at": 2000,
    },
]


def test_an_lmstudio_copy_is_not_aliased_to_a_cached_repo():
    """The alias is only sound for the registered-cache duplicate. An LM Studio
    copy is a different set of files under the same Hub-shaped name, so a cached
    failure must not blacklist it (and its own failure must not blacklist the
    cached repo in the fallback below)."""
    honoured = _run_local_sweep(
        LMSTUDIO_LOOKALIKE,
        failing = [],
        preskipped = ["gguf:unsloth/gemma-3-4b-it-gguf:q4_k_m"],
        cached_repo_dirs = [
            {
                "repoId": "unsloth/gemma-3-4b-it-GGUF",
                "loadPath": "/hub/models--unsloth--gemma-3-4b-it-GGUF",
            },
        ],
    )
    assert honoured["attempted"] == ["gguf|/lmstudio/unsloth/gemma-3-4b-it-GGUF|Q4_K_M"]

    recorded = _run_local_sweep(
        LMSTUDIO_LOOKALIKE,
        failing = [],
        gguf_folder_fail = "/lmstudio/unsloth/gemma-3-4b-it-GGUF",
    )
    assert recorded["skippedKeys"] == ["gguf:/lmstudio/unsloth/gemma-3-4b-it-gguf:q4_k_m"]


def test_mlx_builds_are_only_candidates_on_a_mac():
    """worker.py picks the MLX runner from the detected device, and /validate has
    no MLX preflight, so off a Mac each MLX row spends an attempt and then fails."""
    out = _run(
        "const rows = [\n"
        "  { path: '/models/Qwen3-4B-MLX', display_name: 'Qwen3-4B-MLX' },\n"
        "  { path: '/models/gemma-3-4b-it-MLX-4bit', display_name: 'gemma-3-4b-it-MLX-4bit' },\n"
        "  { id: 'mlx-community/Qwen3-4B', path: '/models/x', display_name: 'Qwen3-4B',\n"
        "    model_id: 'mlx-community/Qwen3-4B' },\n"
        "];\n"
        "const notMlx = [\n"
        "  { path: '/models/Qwen3-4B-GGUF', display_name: 'Qwen3-4B-GGUF' },\n"
        # "MLX" inside a longer word must not match, like the picker's rule.
        "  { path: '/models/MLXplorer-7B', display_name: 'MLXplorer-7B' },\n"
        "];\n"
        "console.log(JSON.stringify({\n"
        "  onLinux: rows.map((o) => helpers.isUnsupportedMlxLocalModel(M(o), false)),\n"
        "  onMac: rows.map((o) => helpers.isUnsupportedMlxLocalModel(M(o), true)),\n"
        "  nonMlxOnLinux: notMlx.map((o) => helpers.isUnsupportedMlxLocalModel(M(o), false)),\n"
        "}));\n"
    )
    assert out == {
        "onLinux": [True, True, True],
        "onMac": [False, False, False],
        "nonMlxOnLinux": [False, False],
    }


def test_the_mlx_pattern_matches_the_pickers():
    """The regex is copied so this module keeps no value imports. Pin the two
    literals against each other so they cannot drift apart."""
    helpers_src = HELPERS.read_text()
    picker_src = _source_path(
        "studio/frontend/src/features/model-picker/components/model-selector/recommended-fit.ts"
    ).read_text()
    literal = "/-MLX(?:$|-)/i"
    assert f"const MLX_RE = {literal};" in helpers_src
    assert f"const MLX_RE = {literal};" in picker_src


MLX_AND_GGUF = [
    {"path": "/scan/Qwen3-4B-MLX", "display_name": "Qwen3-4B-MLX", "updated_at": 3000},
    {"path": "/scan/Llama-3-8B-MLX", "display_name": "Llama-3-8B-MLX", "updated_at": 2500},
    {"path": "/scan/Phi-4-MLX", "display_name": "Phi-4-MLX", "updated_at": 2400},
    {
        "path": "/scan/Qwen3-4B-GGUF",
        "display_name": "Qwen3-4B-GGUF",
        "model_format": "gguf",
        "updated_at": 1000,
    },
]


def test_the_sweep_skips_mlx_rows_off_a_mac():
    """Three MLX directories sorted ahead of a loadable GGUF would otherwise
    spend the whole three-attempt budget before reaching it."""
    out = _run_local_sweep(MLX_AND_GGUF, failing = [], is_mac = False)

    assert out["attempted"] == ["gguf|/scan/Qwen3-4B-GGUF|Q4_K_M"]
    assert out["loadAttempts"] == 1
    assert out["loaded"] is True


def test_the_sweep_keeps_mlx_rows_on_a_mac():
    """On a Mac they are the fastest local option, so ordering is untouched."""
    out = _run_local_sweep(MLX_AND_GGUF, failing = [], is_mac = True)

    assert out["attempted"] == ["model|/scan/Qwen3-4B-MLX|"]
    assert out["loaded"] is True


def test_a_remembered_mlx_model_is_not_restored_off_a_mac():
    """The record lives client side while the host is server side, so the same
    browser profile can point at a Linux server with the path still present (a
    shared models mount). The remembered tier runs before the sweep, so without
    the filter it spends an attempt the sweep's own filter never sees."""
    off_mac = _run_remembered_local(kind = "model", name = "Qwen3-4B-MLX")
    assert off_mac["loaded"] is False
    assert off_mac["skippedKeys"] == []
    assert off_mac["hadNonTrustFailure"] is False

    on_mac = _run_remembered_local(kind = "model", name = "Qwen3-4B-MLX", is_mac = True)
    # Reached on a Mac, where the stubbed load throws and is recorded.
    assert on_mac["hadNonTrustFailure"] is True


# The same repo id in a second cache is a different copy on disk.
SECOND_CACHE_ROWS = [
    {
        "id": "/hubA/models--Org--Foo-GGUF/snapshots/rev0",
        "path": "/hubA/models--Org--Foo-GGUF/snapshots/rev0",
        "model_id": "Org/Foo-GGUF",
        "display_name": "Foo-GGUF",
        "source": "custom",
        "active_cache": False,
        "model_format": "gguf",
        "updated_at": 2000,
    },
    {
        "id": "/hubB/models--Org--Foo-GGUF/snapshots/rev0",
        "path": "/hubB/models--Org--Foo-GGUF/snapshots/rev0",
        "model_id": "Org/Foo-GGUF",
        "display_name": "Foo-GGUF",
        "source": "custom",
        "active_cache": False,
        "model_format": "gguf",
        "updated_at": 1000,
    },
]


def test_only_the_cached_copy_is_aliased_not_a_same_named_repo_elsewhere():
    """Both cached listings collapse copies by repo id, so a cached failure names
    one cache. The registered row from that cache is skipped; the healthy copy in
    the other cache must still be tried."""
    out = _run_local_sweep(
        SECOND_CACHE_ROWS,
        failing = [],
        preskipped = ["gguf:org/foo-gguf:q4_k_m"],
        cached_repo_dirs = [
            {"repoId": "Org/Foo-GGUF", "loadPath": "/hubA/models--Org--Foo-GGUF"},
        ],
    )

    assert out["attempted"] == ["gguf|/hubB/models--Org--Foo-GGUF/snapshots/rev0|Q4_K_M"]
    assert out["loaded"] is True


def test_the_mlx_gate_reads_a_capability_not_the_os_family():
    """/api/health derives device_type from sys.platform, so an Intel Mac and an
    Apple Silicon Mac with a broken MLX stack both report "mac" while
    detect_hardware falls through to CPU and records why. The adapter has to key
    off that reason, not off the OS."""
    src = ADAPTER.read_text()
    gate = _slice(src, "  const hostRunsMlx =", ";\n")
    assert 'platform.deviceType === "mac"' in gate
    assert 'platform.chatOnlyReason !== "mlx_unavailable"' in gate
    assert 'platform.chatOnlyReason !== "intel_mac"' in gate
    # Both reason strings come from hardware.py's CPU fallback.
    hardware = _source_path("studio/backend/utils/hardware/hardware.py").read_text()
    assert 'CHAT_ONLY_REASON = "mlx_unavailable"' in hardware
    assert 'CHAT_ONLY_REASON = "intel_mac"' in hardware


def test_the_cached_safetensors_listing_carries_its_cache_path():
    """The alias matches a local row against the cache dir its cached candidate
    was read from, so the safetensors listing has to report one like the GGUF
    listing does, or the alias is inert for the kind: "model" tier."""
    routes = _source_path("studio/backend/routes/models.py").read_text()
    body = _slice(
        routes,
        "async def list_cached_models(",
        '@router.delete("/delete-cached")',
    )
    assert '"cache_path": str(repo_info.repo_path),' in body


def _run_cached_gguf_tier(*, fail_load: bool, fail_variants: bool = False):
    """Run the REAL cached-GGUF tier over one repo and report what it recorded."""
    src = ADAPTER.read_text()
    key = _slice(src, "function autoLoadCandidateKey(", "\n}\n") + "\n}\n"
    tier = _slice(
        src,
        "    // GGUF first: smallest-total-size repo",
        "    // A registered scan folder is a deliberate",
    )
    return _run(
        "type LastLocalModelKind = 'gguf' | 'model';\n"
        f"{key}\n"
        "const MAX_AUTO_LOAD_ATTEMPTS = 3;\n"
        "let loadAttempts = 0;\n"
        "let hadNonTrustFailure = false;\n"
        "const skippedAutoLoadCandidates = new Set<string>();\n"
        "const attempted: string[] = [];\n"
        "function isAutoLoadableGgufVariant(_v: any) { return true; }\n"
        f"const FAIL_VARIANTS = {json.dumps(fail_variants)};\n"
        f"const FAIL_LOAD = {json.dumps(fail_load)};\n"
        "async function listGgufVariants(_id: string, _b?: any, _c?: any) {\n"
        "  if (FAIL_VARIANTS) { throw new Error('network'); }\n"
        "  return { variants: [{ quant: 'Q4_K_M', downloaded: true, size_bytes: 4000 }] };\n"
        "}\n"
        "async function loadAutoLoadCandidate(candidate: any): Promise<boolean> {\n"
        "  attempted.push(`${candidate.id}|${candidate.ggufVariant}`);\n"
        "  loadAttempts += 1;\n"
        "  if (FAIL_LOAD) { throw new Error('llama_model_load: error loading model'); }\n"
        "  return true;\n"
        "}\n"
        "const ggufRepos = [{ repo_id: 'Org/Foo-GGUF', size_bytes: 10,\n"
        "  cache_path: '/hub/models--Org--Foo-GGUF' }];\n"
        "async function cachedGgufTier() {\n"
        f"{tier}\n"
        "  return { loaded: false, blockedByTrustRemoteCode: false };\n"
        "}\n"
        "const result = await cachedGgufTier();\n"
        "console.log(JSON.stringify({\n"
        "  loaded: result.loaded, attempted, loadAttempts, hadNonTrustFailure,\n"
        "  skippedKeys: [...skippedAutoLoadCandidates],\n"
        "}));\n"
    )


def test_a_thrown_cached_gguf_load_is_recorded_for_the_local_sweep():
    """The cached tier runs first. When that repo's cache is a registered scan
    folder the same snapshot reaches the sweep under its absolute path, so a
    throw here has to leave a key the local row's repo-id alias can match."""
    out = _run_cached_gguf_tier(fail_load = True)

    assert out["attempted"] == ["Org/Foo-GGUF|Q4_K_M"]
    assert out["skippedKeys"] == ["gguf:org/foo-gguf:q4_k_m"]
    assert out["hadNonTrustFailure"] is True
    assert out["loaded"] is False


def test_a_failed_variants_lookup_does_not_blacklist_the_cached_repo():
    """No load was attempted, so there is no quant to blame. The local row
    enumerates variants by path and may still succeed."""
    out = _run_cached_gguf_tier(fail_load = False, fail_variants = True)

    assert out["attempted"] == []
    assert out["skippedKeys"] == []
    assert out["hadNonTrustFailure"] is True


def test_a_successful_cached_gguf_load_records_nothing():
    out = _run_cached_gguf_tier(fail_load = False)

    assert out["loaded"] is True
    assert out["skippedKeys"] == []
    assert out["hadNonTrustFailure"] is False


def test_a_remembered_repo_id_only_matches_a_cache_snapshot():
    """A remembered Hub id comes from a cached load. An LM Studio row carries a
    publisher/model-name model_id of the same shape for an independent copy, so
    only the registered cache snapshot may answer to it."""
    out = _run(
        "const cacheRow = M({ id: '/hub/models--Org--Foo/snapshots/rev0',\n"
        "  path: '/hub/models--Org--Foo/snapshots/rev0', model_id: 'Org/Foo',\n"
        "  active_cache: false });\n"
        "const lmStudioRow = M({ id: '/lmstudio/Org/Foo', path: '/lmstudio/Org/Foo',\n"
        "  model_id: 'Org/Foo', source: 'lmstudio' });\n"
        "const find = (rows, id) => helpers.findLocalModel(rows, id)?.id ?? null;\n"
        "console.log(JSON.stringify({\n"
        "  cacheByRepoId: find([cacheRow], 'Org/Foo'),\n"
        "  lmStudioByRepoId: find([lmStudioRow], 'Org/Foo'),\n"
        "  lmStudioByPath: find([lmStudioRow], '/lmstudio/Org/Foo'),\n"
        "  bothPrefersCache: find([lmStudioRow, cacheRow], 'Org/Foo'),\n"
        "}));\n"
    )
    assert out == {
        "cacheByRepoId": "/hub/models--Org--Foo/snapshots/rev0",
        "lmStudioByRepoId": None,
        "lmStudioByPath": "/lmstudio/Org/Foo",
        "bothPrefersCache": "/hub/models--Org--Foo/snapshots/rev0",
    }


def test_the_mlx_test_is_scoped_to_the_entrys_own_segment():
    """A parent directory named like an MLX drive must not condemn every
    checkpoint under it, the same false positive localModelIsGguf guards."""
    out = _run(
        "console.log(JSON.stringify({\n"
        "  parentOnly: helpers.isUnsupportedMlxLocalModel(M({\n"
        "    path: '/mnt/my-MLX-models/Qwen3-4B', display_name: 'Qwen3-4B' }), false),\n"
        "  ownName: helpers.isUnsupportedMlxLocalModel(M({\n"
        "    path: '/mnt/models/Qwen3-4B-MLX', display_name: 'Qwen3-4B-MLX' }), false),\n"
        "  byRepoId: helpers.isUnsupportedMlxLocalModel(M({\n"
        "    path: '/lmstudio/mlx-community/Qwen3-4B', display_name: 'Qwen3-4B',\n"
        "    model_id: 'mlx-community/Qwen3-4B' }), false),\n"
        "}));\n"
    )
    assert out == {"parentOnly": False, "ownName": True, "byRepoId": True}

