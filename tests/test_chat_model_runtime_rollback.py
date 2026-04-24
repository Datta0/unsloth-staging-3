import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TS_PATH = (
    REPO_ROOT
    / "studio/frontend/src/features/chat/hooks/use-chat-model-runtime.ts"
)


@pytest.fixture(scope="module")
def source():
    if not TS_PATH.exists():
        pytest.skip(f"source missing: {TS_PATH}")
    return TS_PATH.read_text()


def _extract_decl(source, name):
    pattern = rf"const\s+{re.escape(name)}\s*=\s*([\s\S]*?);"
    m = re.search(pattern, source)
    assert m, f"declaration not found: {name}"
    return m.group(1)


def test_target_max_seq_length_is_user_slider(source):
    body = _extract_decl(source, "maxSeqLength")
    assert "params.maxSeqLength" in body
    assert "previousModel" not in body
    assert "previousVariant" not in body
    assert "ggufContextLength" not in body
    assert "customContextLength" not in body


def test_previous_gguf_is_detected_via_variant_and_suffix(source):
    body = _extract_decl(source, "previousIsGguf")
    assert "previousModel" in body and "isGguf" in body
    assert "previousVariant" in body
    assert ".gguf" in body.lower()


def test_rollback_max_seq_length_uses_zero_sentinel(source):
    body = _extract_decl(source, "rollbackMaxSeqLength")
    assert "ggufContextLength" in body
    assert re.search(r"\?\?\s*0\b", body), "expected 0 sentinel fallback"
    assert "DEFAULT_INFERENCE_PARAMS" not in body, (
        "4096 fallback would truncate GGUF context on rollback"
    )
    assert "customContextLength" not in body, (
        "pending slider value must not override confirmed GGUF context in rollback"
    )


def test_default_inference_params_value_import_removed(source):
    pre_body = source.split("export function useChatModelRuntime")[0]
    imports = re.search(
        r"from \"\.\./types/runtime\";",
        pre_body,
    )
    assert imports, "runtime types import not found"
    preamble = pre_body[: imports.end()]
    assert "DEFAULT_INFERENCE_PARAMS" not in preamble, (
        "value import should be removed once the 4096 fallback is gone"
    )


def test_rollback_loadmodel_passes_rollback_max_seq_length(source):
    m = re.search(
        r"if \(previousWasUnloaded && previousCheckpoint\) \{"
        r"[\s\S]*?loadModel\(\{([\s\S]*?)\}\)",
        source,
    )
    assert m, "rollback loadModel call not found"
    call_args = m.group(1)
    assert "max_seq_length: rollbackMaxSeqLength" in call_args
    assert re.search(r"max_seq_length:\s*maxSeqLength\b", call_args) is None, (
        "rollback must not use the target maxSeqLength"
    )


def test_target_validate_and_load_still_use_user_max_seq_length(source):
    m = re.search(
        r"validateModel\(\{([\s\S]*?)\}\)",
        source,
    )
    assert m
    assert "max_seq_length: maxSeqLength" in m.group(1)
    m2 = re.search(
        r"const\s+effectiveMaxSeqLength\s*=([\s\S]*?);",
        source,
    )
    assert m2
    tail = m2.group(1).strip().splitlines()[-1]
    assert "maxSeqLength" in tail, (
        "non-GGUF effectiveMaxSeqLength must fall back to the user slider value"
    )


def test_trust_remote_code_reuses_captured_state_snapshot(source):
    m = re.search(
        r"const\s+previousModelRequiresTrustRemoteCode\s*=\s*([\s\S]*?);",
        source,
    )
    assert m
    body = m.group(1)
    assert "stateBeforeUnload" in body
    assert "getState()" not in body
