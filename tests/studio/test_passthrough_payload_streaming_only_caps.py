"""
_build_passthrough_payload is shared by streaming and non-streaming
OpenAI/Anthropic passthrough paths. Studio's max_tokens=4096 and
t_max_predict_ms defaults must only apply when stream=True; non-streaming
passthrough must preserve pre-PR behavior (forward max_tokens only if the
caller set it, never inject t_max_predict_ms or stream_options).
"""

from __future__ import annotations

import ast
from pathlib import Path


ROUTES_PATH = (
    Path(__file__).resolve().parents[2]
    / "studio"
    / "backend"
    / "routes"
    / "inference.py"
)
LLAMA_PATH = (
    Path(__file__).resolve().parents[2]
    / "studio"
    / "backend"
    / "core"
    / "inference"
    / "llama_cpp.py"
)

_ROUTES_SRC = ROUTES_PATH.read_text()
_ROUTES_TREE = ast.parse(_ROUTES_SRC)
_LLAMA_SRC = LLAMA_PATH.read_text()
_LLAMA_TREE = ast.parse(_LLAMA_SRC)


def _load_helper_module():
    fn_src = None
    for n in _ROUTES_TREE.body:
        if isinstance(n, ast.FunctionDef) and n.name == "_build_passthrough_payload":
            fn_src = ast.get_source_segment(_ROUTES_SRC, n)
            break
    assert fn_src is not None, "_build_passthrough_payload not found"

    const_chunks = []
    wanted = {"_DEFAULT_MAX_TOKENS", "_DEFAULT_T_MAX_PREDICT_MS"}
    for n in _LLAMA_TREE.body:
        if isinstance(n, ast.Assign):
            names = [t.id for t in n.targets if isinstance(t, ast.Name)]
            if any(name in wanted for name in names):
                const_chunks.append(ast.get_source_segment(_LLAMA_SRC, n))

    mod = {}
    exec("\n\n".join(const_chunks + [fn_src]), mod)
    return mod


def _call_helper(mod, *, max_tokens, stream):
    return mod["_build_passthrough_payload"](
        [],
        [],
        0.6,
        0.95,
        20,
        max_tokens,
        stream,
    )


def test_non_streaming_no_max_tokens_forwards_no_caps():
    m = _load_helper_module()
    body = _call_helper(m, max_tokens=None, stream=False)
    assert "max_tokens" not in body, (
        "non-streaming passthrough with max_tokens=None must NOT inject a "
        "default cap; pre-PR behavior forwarded the request uncapped and "
        "external OpenAI/Anthropic clients rely on that."
    )
    assert "t_max_predict_ms" not in body, (
        "non-streaming must never carry the Studio wall-clock backstop; it "
        "would silently cut off long non-streaming generations."
    )
    assert "stream_options" not in body


def test_non_streaming_with_explicit_max_tokens_is_forwarded_verbatim():
    m = _load_helper_module()
    body = _call_helper(m, max_tokens=1234, stream=False)
    assert body["max_tokens"] == 1234
    assert "t_max_predict_ms" not in body
    assert "stream_options" not in body


def test_streaming_no_max_tokens_applies_studio_defaults():
    m = _load_helper_module()
    body = _call_helper(m, max_tokens=None, stream=True)
    assert body["max_tokens"] == m["_DEFAULT_MAX_TOKENS"]
    assert body["t_max_predict_ms"] == m["_DEFAULT_T_MAX_PREDICT_MS"]
    assert body["stream_options"] == {"include_usage": True}


def test_streaming_with_explicit_max_tokens_preserves_user_value_and_timeout():
    m = _load_helper_module()
    body = _call_helper(m, max_tokens=256, stream=True)
    assert body["max_tokens"] == 256
    assert body["t_max_predict_ms"] == m["_DEFAULT_T_MAX_PREDICT_MS"]
    assert body["stream_options"] == {"include_usage": True}


def test_streaming_default_max_tokens_is_4096():
    m = _load_helper_module()
    assert m["_DEFAULT_MAX_TOKENS"] == 4096


def test_streaming_default_wall_clock_gives_at_least_thirty_minutes():
    m = _load_helper_module()
    assert m["_DEFAULT_T_MAX_PREDICT_MS"] >= 1_800_000, (
        f"_DEFAULT_T_MAX_PREDICT_MS must be >= 30 min to cover slow-CPU "
        f"generations (2-3 t/s x 4096 tokens); got "
        f"{m['_DEFAULT_T_MAX_PREDICT_MS']}"
    )
