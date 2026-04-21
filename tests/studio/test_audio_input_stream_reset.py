"""
audio_input_stream is the non-GGUF (Unsloth) streaming path for audio
input; it shares the same `backend` object as stream_chunks. Every
termination path must call backend.reset_generation_state() so the
KV-cache / GPU state is clean before the next request. stream_chunks
already does this on all four paths; audio_input_stream was missing the
calls.
"""

from __future__ import annotations

import ast
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


def _find_async_gen(name):
    for node in ast.walk(_TREE):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == name:
            return node
    raise AssertionError(f"async generator {name!r} not found")


def _calls_backend_reset(node) -> bool:
    src = ast.unparse(node)
    return "backend.reset_generation_state()" in src


def _first_matching_if(fn, predicate):
    for sub in ast.walk(fn):
        if isinstance(sub, ast.If) and predicate(sub.test):
            return sub
    return None


def _first_matching_except(fn, predicate):
    for sub in ast.walk(fn):
        if isinstance(sub, ast.ExceptHandler) and sub.type is not None and predicate(sub.type):
            return sub
    return None


def test_audio_input_stream_cancel_branch_calls_backend_reset():
    fn = _find_async_gen("audio_input_stream")

    def _is_cancel_check(t):
        return (
            isinstance(t, ast.Call)
            and isinstance(t.func, ast.Attribute)
            and t.func.attr == "is_set"
            and isinstance(t.func.value, ast.Name)
            and t.func.value.id == "cancel_event"
        )

    branch = _first_matching_if(fn, _is_cancel_check)
    assert branch is not None, "cancel_event.is_set() branch not found"
    body_src = "\n".join(ast.unparse(s) for s in branch.body)
    assert "backend.reset_generation_state()" in body_src, (
        "audio_input_stream cancel branch must call "
        "backend.reset_generation_state() before breaking; matches "
        "stream_chunks and prevents KV-cache drift on the next request."
    )


def test_audio_input_stream_disconnect_branch_calls_backend_reset():
    fn = _find_async_gen("audio_input_stream")

    def _is_disconnected_check(t):
        inner = t.value if isinstance(t, ast.Await) else t
        return (
            isinstance(inner, ast.Call)
            and isinstance(inner.func, ast.Attribute)
            and inner.func.attr == "is_disconnected"
        )

    branch = _first_matching_if(fn, _is_disconnected_check)
    assert branch is not None, "is_disconnected() branch not found"
    body_src = "\n".join(ast.unparse(s) for s in branch.body)
    assert "backend.reset_generation_state()" in body_src, (
        "audio_input_stream is_disconnected() branch must call "
        "backend.reset_generation_state() before returning."
    )


def test_audio_input_stream_cancelled_error_handler_calls_backend_reset():
    fn = _find_async_gen("audio_input_stream")

    def _is_cancelled_error(t):
        return (
            isinstance(t, ast.Attribute)
            and t.attr == "CancelledError"
            and isinstance(t.value, ast.Name)
            and t.value.id == "asyncio"
        )

    handler = _first_matching_except(fn, _is_cancelled_error)
    assert handler is not None, "except asyncio.CancelledError handler not found"
    body_src = "\n".join(ast.unparse(s) for s in handler.body)
    assert "backend.reset_generation_state()" in body_src, (
        "audio_input_stream except asyncio.CancelledError must call "
        "backend.reset_generation_state() before re-raising."
    )


def test_audio_input_stream_exception_handler_calls_backend_reset():
    fn = _find_async_gen("audio_input_stream")

    def _is_plain_exception(t):
        return isinstance(t, ast.Name) and t.id == "Exception"

    handler = _first_matching_except(fn, _is_plain_exception)
    assert handler is not None, "except Exception handler not found"
    body_src = "\n".join(ast.unparse(s) for s in handler.body)
    assert "backend.reset_generation_state()" in body_src, (
        "audio_input_stream except Exception must call "
        "backend.reset_generation_state() before emitting the error chunk."
    )


def test_audio_input_stream_reset_call_count_matches_stream_chunks_pattern():
    audio_fn = _find_async_gen("audio_input_stream")
    stream_chunks_fn = None
    for node in ast.walk(_TREE):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "stream_chunks":
            stream_chunks_fn = node
            break
    assert stream_chunks_fn is not None, "stream_chunks not found"

    audio_src = ast.unparse(audio_fn)
    sc_src = ast.unparse(stream_chunks_fn)
    audio_resets = audio_src.count("backend.reset_generation_state()")
    sc_resets = sc_src.count("backend.reset_generation_state()")
    assert audio_resets >= 4, (
        f"audio_input_stream must reset backend on all 4 termination paths "
        f"(cancel/disconnect/CancelledError/Exception); got {audio_resets}"
    )
    assert audio_resets <= sc_resets, (
        f"stream_chunks has {sc_resets} reset calls, audio_input_stream has "
        f"{audio_resets}; audio should not exceed the canonical pattern."
    )
