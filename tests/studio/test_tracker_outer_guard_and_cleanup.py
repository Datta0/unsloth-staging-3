"""
Outer-guard and cleanup safety nets for the streaming handlers.

Three classes of leak/state regressions:

1) _tracker.__enter__() runs in the sync handler body, __exit__ in the
   async generator's finally. If StreamingResponse construction raises or
   the returned generator is GC'd before iteration starts, the registry
   entry leaks. _openai_passthrough_stream demonstrates the fix: wrap
   `return StreamingResponse(...)` with `try: ... except BaseException:
   _tracker.__exit__(None, None, None); raise`. All five streaming sites
   must use this same pattern (four OpenAI-endpoint generators plus
   _anthropic_passthrough_stream, covered structurally here).

2) audio_input_stream's cancel-by-POST path must flush the Unsloth
   InferenceBackend's generation state -- it iterates through
   backend.generate_audio_input_response / generate_whisper_response,
   which hold the _gen_lock and leave GPU / KV state dirty if not
   reset. stream_chunks does this on both cancel_event.is_set() and
   request.is_disconnected(); audio_input_stream now matches.

3) _openai_passthrough_stream's outer `except BaseException` guard must
   close the httpx.AsyncClient in addition to calling _tracker.__exit__.
   asyncio.CancelledError during `await client.send(...)` is a
   BaseException that bypasses `except httpx.RequestError`; without
   aclose() the client file descriptor leaks.
"""

from __future__ import annotations

import ast
from pathlib import Path


SOURCE = (
    Path(__file__).resolve().parents[2]
    / "studio"
    / "backend"
    / "routes"
    / "inference.py"
)
_SRC = SOURCE.read_text()
_TREE = ast.parse(_SRC)


def _find_function(name):
    for n in ast.walk(_TREE):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name:
            return n
    raise AssertionError(f"{name!r} not found")


def _collect_enter_exit_pairs(fn):
    """Return list of (enter_stmt_index, enclosing_try_or_None) for each
    `_tracker.__enter__()` call appearing directly in fn.body or in a
    nested `if payload.stream:` branch. The caller uses this to check
    that a matching `try: return StreamingResponse(...) except
    BaseException: _tracker.__exit__(...); raise` exists nearby."""
    # Traverse all statements, tracking the stack of enclosing nodes.
    pairs = []

    def visit(stmts, parent_key = ""):
        for idx, stmt in enumerate(stmts):
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                call = stmt.value
                if (
                    isinstance(call.func, ast.Attribute)
                    and call.func.attr == "__enter__"
                    and isinstance(call.func.value, ast.Name)
                    and call.func.value.id == "_tracker"
                ):
                    pairs.append((stmts, idx))
            # Recurse into control-flow children that can contain both
            # __enter__ and the paired `return StreamingResponse(...)`.
            for attr in ("body", "orelse", "finalbody"):
                children = getattr(stmt, attr, None)
                if isinstance(children, list):
                    visit(children, parent_key + f".{attr}")
            if isinstance(stmt, ast.Try):
                for h in stmt.handlers:
                    visit(h.body, parent_key + ".handler")

    visit(fn.body)
    return pairs


def _find_following_try_with_baseexception(stmts, start_idx):
    # Look at statements after the _tracker.__enter__() call within the
    # same block for a Try whose handler is `except BaseException:` and
    # whose body contains `_tracker.__exit__` and `raise`.
    for j in range(start_idx + 1, len(stmts)):
        stmt = stmts[j]
        if isinstance(stmt, ast.Try):
            for handler in stmt.handlers:
                t = handler.type
                if (
                    isinstance(t, ast.Name)
                    and t.id == "BaseException"
                ):
                    body_src = "\n".join(ast.unparse(s) for s in handler.body)
                    if "_tracker.__exit__" in body_src and "raise" in body_src:
                        return True
            # Try found but no BaseException handler -- fail fast
            return False
    return False


# ── Outer-guard coverage ─────────────────────────────────────


def test_openai_chat_completions_streaming_sites_have_outer_guard():
    fn = _find_function("openai_chat_completions")
    pairs = _collect_enter_exit_pairs(fn)
    # openai_chat_completions contains four streaming sites:
    # audio_input_stream, gguf_tool_stream, gguf_stream_chunks, stream_chunks.
    # Each calls _tracker.__enter__() before its return StreamingResponse.
    assert len(pairs) >= 4, (
        f"expected >=4 _tracker.__enter__() sites in openai_chat_completions, "
        f"got {len(pairs)}"
    )
    for stmts, idx in pairs:
        assert _find_following_try_with_baseexception(stmts, idx), (
            "every _tracker.__enter__() call in openai_chat_completions must "
            "be immediately followed by a `try: return StreamingResponse(...) "
            "except BaseException: _tracker.__exit__(None, None, None); raise` "
            "so a construction failure does not leak the registry entry"
        )


def test_anthropic_passthrough_stream_has_outer_guard_on_return():
    fn = _find_function("_anthropic_passthrough_stream")
    src = ast.unparse(fn)
    # Search for the specific idiom at the tail of the function.
    assert (
        "except BaseException:" in src
        and "_tracker.__exit__(None, None, None)" in src
    ), (
        "_anthropic_passthrough_stream must wrap its return StreamingResponse "
        "in try/except BaseException to release the registry entry on "
        "construction failure"
    )


# ── audio_input_stream reset on cancel ───────────────────────


def test_audio_input_stream_resets_backend_on_both_cancel_branches():
    fn = _find_function("openai_chat_completions")
    gen = None
    for sub in ast.walk(fn):
        if isinstance(sub, ast.AsyncFunctionDef) and sub.name == "audio_input_stream":
            gen = sub
            break
    assert gen is not None

    # Find the while-True loop inside audio_input_stream.
    while_loop = None
    for sub in ast.walk(gen):
        if isinstance(sub, ast.While):
            while_loop = sub
            break
    assert while_loop is not None, "audio_input_stream must drive a while loop"

    cancel_branch_resets = False
    disconnect_branch_resets = False
    for stmt in while_loop.body:
        if not isinstance(stmt, ast.If):
            continue
        t = stmt.test
        body_src = "\n".join(ast.unparse(s) for s in stmt.body)
        # cancel_event.is_set() branch
        if (
            isinstance(t, ast.Call)
            and isinstance(t.func, ast.Attribute)
            and t.func.attr == "is_set"
            and isinstance(t.func.value, ast.Name)
            and t.func.value.id == "cancel_event"
        ):
            if "backend.reset_generation_state()" in body_src:
                cancel_branch_resets = True
        # await request.is_disconnected() branch
        if isinstance(t, ast.Await):
            inner = t.value
            if (
                isinstance(inner, ast.Call)
                and isinstance(inner.func, ast.Attribute)
                and inner.func.attr == "is_disconnected"
            ):
                if "backend.reset_generation_state()" in body_src:
                    disconnect_branch_resets = True

    assert cancel_branch_resets, (
        "audio_input_stream's `if cancel_event.is_set():` branch must call "
        "backend.reset_generation_state() -- matches stream_chunks and "
        "prevents the Unsloth InferenceBackend generation lock from being "
        "held after cancel-via-POST"
    )
    assert disconnect_branch_resets, (
        "audio_input_stream's `if await request.is_disconnected():` branch "
        "must also call backend.reset_generation_state() so native proxy "
        "disconnects get the same cleanup"
    )


# ── _openai_passthrough_stream closes client on BaseException ──


def test_openai_passthrough_outer_guard_closes_client():
    fn = _find_function("_openai_passthrough_stream")

    # client must be initialized to None before the outer try so the
    # BaseException handler can check `if client is not None`.
    init_found = False
    for stmt in fn.body:
        if (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
            and stmt.targets[0].id == "client"
            and isinstance(stmt.value, ast.Constant)
            and stmt.value.value is None
        ):
            init_found = True
            break
    assert init_found, (
        "_openai_passthrough_stream must initialise `client = None` at the "
        "top of the function so the BaseException handler can safely aclose "
        "it without NameError if send() fails before client construction"
    )

    # The outer except BaseException must call `await client.aclose()`
    # guarded by `if client is not None`.
    outer_try = None
    for stmt in fn.body:
        if isinstance(stmt, ast.Try):
            for handler in stmt.handlers:
                if isinstance(handler.type, ast.Name) and handler.type.id == "BaseException":
                    outer_try = handler
                    break
        if outer_try is not None:
            break
    assert outer_try is not None, (
        "_openai_passthrough_stream must have a top-level except BaseException"
    )
    handler_src = "\n".join(ast.unparse(s) for s in outer_try.body)
    assert "client is not None" in handler_src, (
        "BaseException handler must guard aclose with `if client is not None`"
    )
    assert "await client.aclose()" in handler_src, (
        "BaseException handler must call `await client.aclose()` -- "
        "asyncio.CancelledError during client.send() bypasses the inner "
        "except httpx.RequestError and would otherwise leak the AsyncClient"
    )
    assert "_tracker.__exit__(None, None, None)" in handler_src
    assert "raise" in handler_src
