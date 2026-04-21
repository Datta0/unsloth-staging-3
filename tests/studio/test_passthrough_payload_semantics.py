"""
Guards the /v1 OpenAI and Anthropic passthrough semantics.

_build_passthrough_payload forwards external-SDK requests to llama-server
verbatim. The PR originally injected max_tokens=_DEFAULT_MAX_TOKENS and
t_max_predict_ms=_DEFAULT_T_MAX_PREDICT_MS unconditionally, which silently
imposed a 4096-token cap and a 10-minute wall-clock stop on every passthrough
request that omitted max_tokens -- a breaking change for any external caller
that relied on the upstream server-default behaviour.

These tests exercise the behavioural contract by importing and executing the
builder (no FastAPI stack required) so the defaults cannot regress back into
the body for passthrough callers.
"""

from __future__ import annotations

import ast
import sys
import textwrap
import types
from pathlib import Path


SOURCE_PATH = (
    Path(__file__).resolve().parents[2]
    / "studio"
    / "backend"
    / "routes"
    / "inference.py"
)
_SRC = SOURCE_PATH.read_text()
_TREE = ast.parse(_SRC)


def _find_function(name):
    for node in ast.walk(_TREE):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"{name!r} not found")


def _load_builder():
    fn = _find_function("_build_passthrough_payload")
    module = types.ModuleType("_build_passthrough_module")
    # Constants referenced in the body if the fix were reverted -- make them
    # nonzero so any regression shows up in the assertions below.
    module.__dict__["_DEFAULT_MAX_TOKENS"] = 4096
    module.__dict__["_DEFAULT_T_MAX_PREDICT_MS"] = 600_000
    exec(compile(ast.Module(body = [fn], type_ignores = []), str(SOURCE_PATH), "exec"), module.__dict__)
    sys.modules["_build_passthrough_module"] = module
    return module._build_passthrough_payload


def test_omitted_max_tokens_streaming_does_not_inject_defaults():
    build = _load_builder()
    body = build(
        openai_messages = [{"role": "user", "content": "hi"}],
        openai_tools = None,
        temperature = 0.6,
        top_p = 0.95,
        top_k = 20,
        max_tokens = None,
        stream = True,
    )
    assert "max_tokens" not in body, (
        "passthrough must NOT inject a default max_tokens when the external "
        "caller omits it; imposes a silent 4096-token cap on /v1 callers"
    )
    assert "t_max_predict_ms" not in body, (
        "passthrough must NOT inject t_max_predict_ms; imposes a silent 10-min "
        "wall-clock stop on /v1 callers that never asked for one"
    )


def test_omitted_max_tokens_non_streaming_does_not_inject_defaults():
    # The pre-fix diff produced the regression on `stream=False` too because
    # the injection sat below `if stream: ...` -- guard that specifically.
    build = _load_builder()
    body = build(
        openai_messages = [{"role": "user", "content": "hi"}],
        openai_tools = None,
        temperature = 0.6,
        top_p = 0.95,
        top_k = 20,
        max_tokens = None,
        stream = False,
    )
    assert "max_tokens" not in body
    assert "t_max_predict_ms" not in body
    assert "stream_options" not in body


def test_explicit_max_tokens_is_forwarded_verbatim():
    build = _load_builder()
    body = build(
        openai_messages = [{"role": "user", "content": "hi"}],
        openai_tools = None,
        temperature = 0.6,
        top_p = 0.95,
        top_k = 20,
        max_tokens = 123,
        stream = True,
    )
    assert body["max_tokens"] == 123, (
        "caller-supplied max_tokens must flow through untouched to llama-server"
    )
    assert "t_max_predict_ms" not in body


def test_stop_and_sampler_kwargs_still_flow_through():
    # Regression guard: the fix removed two lines; make sure the surrounding
    # stop / min_p / repetition_penalty / presence_penalty branches survived.
    build = _load_builder()
    body = build(
        openai_messages = [{"role": "user", "content": "hi"}],
        openai_tools = [{"type": "function", "function": {"name": "noop"}}],
        temperature = 0.3,
        top_p = 0.9,
        top_k = 40,
        max_tokens = None,
        stream = True,
        stop = ["</s>"],
        min_p = 0.05,
        repetition_penalty = 1.2,
        presence_penalty = 0.5,
        tool_choice = "auto",
    )
    assert body["stop"] == ["</s>"]
    assert body["min_p"] == 0.05
    # llama-server's native field name is repeat_penalty
    assert body["repeat_penalty"] == 1.2
    assert body["presence_penalty"] == 0.5
    assert body["tool_choice"] == "auto"
    assert body["tools"] == [{"type": "function", "function": {"name": "noop"}}]


def test_builder_does_not_reference_default_constants_after_fix():
    # Structural guard: if someone re-adds the unconditional injection we
    # want to catch it even before import-level execution.
    fn = _find_function("_build_passthrough_payload")
    src = ast.unparse(fn)
    assert "_DEFAULT_MAX_TOKENS" not in src, (
        "_build_passthrough_payload must not fall back to _DEFAULT_MAX_TOKENS; "
        "that reimposes the /v1 cap regression"
    )
    assert "_DEFAULT_T_MAX_PREDICT_MS" not in src, (
        "_build_passthrough_payload must not inject _DEFAULT_T_MAX_PREDICT_MS"
    )
