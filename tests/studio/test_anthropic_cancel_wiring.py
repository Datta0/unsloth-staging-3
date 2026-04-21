"""
Cancel-registry wiring for every Anthropic streaming path.

The frontend always sends a fresh per-run cancel_id alongside session_id.
The stop POST to /api/inference/cancel preferentially matches cancel_id and
falls back to session_id only when cancel_id is absent. If an Anthropic
streaming handler does not register cancel_id with _TrackedCancel then
_cancel_by_cancel_id_or_stash stashes the uuid and lets it expire silently
-- the stop button becomes a no-op for that endpoint.

Covers:
  - AnthropicMessagesRequest exposes an Optional[str] cancel_id field.
  - All three streaming helpers (_anthropic_passthrough_stream,
    _anthropic_tool_stream, _anthropic_plain_stream) accept cancel_id and
    register it with _TrackedCancel.
  - Each helper installs _tracker.__exit__ in a generator-level finally.
  - Each helper wraps `return StreamingResponse(...)` with an outer
    try/except BaseException that calls _tracker.__exit__ and re-raises.
  - Each helper's main loop checks cancel_event.is_set() at the top.
  - The anthropic_messages handler threads payload.cancel_id and
    payload.session_id into all three call sites.
"""

from __future__ import annotations

import ast
from pathlib import Path


ROUTES = (
    Path(__file__).resolve().parents[2]
    / "studio"
    / "backend"
    / "routes"
    / "inference.py"
)
MODELS = (
    Path(__file__).resolve().parents[2]
    / "studio"
    / "backend"
    / "models"
    / "inference.py"
)

_ROUTES_SRC = ROUTES.read_text()
_MODELS_SRC = MODELS.read_text()
_ROUTES_TREE = ast.parse(_ROUTES_SRC)
_MODELS_TREE = ast.parse(_MODELS_SRC)


def _find(tree, name, cls = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
    for n in ast.walk(tree):
        if isinstance(n, cls) and n.name == name:
            return n
    raise AssertionError(f"{name!r} not found")


def _has_tracker_call_with(fn, *keys):
    """True iff fn contains `_TrackedCancel(cancel_event, <keys in order>)`."""
    for sub in ast.walk(fn):
        if not (isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name)):
            continue
        if sub.func.id != "_TrackedCancel":
            continue
        if len(sub.args) != 1 + len(keys):
            continue
        if not (isinstance(sub.args[0], ast.Name) and sub.args[0].id == "cancel_event"):
            continue
        ok = True
        for arg, expected in zip(sub.args[1:], keys):
            if not (isinstance(arg, ast.Name) and arg.id == expected):
                ok = False
                break
        if ok:
            return True
    return False


def _finally_calls_tracker_exit(fn):
    for sub in ast.walk(fn):
        if not (isinstance(sub, ast.Try) and sub.finalbody):
            continue
        for stmt in sub.finalbody:
            if not isinstance(stmt, ast.Expr):
                continue
            call = stmt.value
            if (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr == "__exit__"
                and isinstance(call.func.value, ast.Name)
                and call.func.value.id.startswith("_tracker")
            ):
                return True
    return False


def _has_outer_baseexception_guard(fn):
    """True iff the function body contains a top-level try whose handler is
    `except BaseException:` and calls _tracker.__exit__ then `raise`."""
    for stmt in fn.body:
        if not isinstance(stmt, ast.Try):
            continue
        for handler in stmt.handlers:
            t = handler.type
            if not (isinstance(t, ast.Name) and t.id == "BaseException"):
                continue
            body_src = "\n".join(ast.unparse(s) for s in handler.body)
            if "_tracker.__exit__" in body_src and "raise" in body_src:
                return True
    return False


def _loop_checks_cancel_event(fn):
    for sub in ast.walk(fn):
        if not isinstance(sub, (ast.While, ast.For, ast.AsyncFor)):
            continue
        for stmt in ast.walk(sub):
            if not isinstance(stmt, ast.If):
                continue
            t = stmt.test
            if (
                isinstance(t, ast.Call)
                and isinstance(t.func, ast.Attribute)
                and t.func.attr == "is_set"
                and isinstance(t.func.value, ast.Name)
                and t.func.value.id == "cancel_event"
            ):
                return True
    return False


# ── Field on the request model ───────────────────────────────


def test_anthropic_request_has_optional_cancel_id_field():
    cls = _find(_MODELS_TREE, "AnthropicMessagesRequest")
    for n in cls.body:
        if (
            isinstance(n, ast.AnnAssign)
            and isinstance(n.target, ast.Name)
            and n.target.id == "cancel_id"
        ):
            src = ast.unparse(n.annotation)
            assert "Optional" in src and "str" in src, (
                f"cancel_id must be Optional[str]; got annotation {src!r}"
            )
            return
    raise AssertionError(
        "AnthropicMessagesRequest must expose a cancel_id field so Anthropic "
        "callers can participate in the per-run cancel registry"
    )


# ── Each streaming helper is wired ───────────────────────────


def test_anthropic_passthrough_stream_registers_cancel_id():
    fn = _find(_ROUTES_TREE, "_anthropic_passthrough_stream")
    params = {a.arg for a in fn.args.args + fn.args.kwonlyargs}
    assert "cancel_id" in params, "passthrough stream must accept cancel_id"
    assert "session_id" in params
    assert _has_tracker_call_with(
        fn, "cancel_id", "session_id", "message_id"
    ), (
        "_anthropic_passthrough_stream must register _TrackedCancel("
        "cancel_event, cancel_id, session_id, message_id)"
    )
    assert _finally_calls_tracker_exit(fn)
    assert _has_outer_baseexception_guard(fn), (
        "passthrough stream must wrap the return StreamingResponse in an "
        "outer try/except BaseException that calls _tracker.__exit__ so the "
        "registry entry does not leak if construction raises"
    )


def test_anthropic_tool_stream_registers_and_checks_cancel_event():
    fn = _find(_ROUTES_TREE, "_anthropic_tool_stream")
    params = {a.arg for a in fn.args.args + fn.args.kwonlyargs}
    assert {"cancel_id", "session_id"} <= params
    assert _has_tracker_call_with(
        fn, "cancel_id", "session_id", "message_id"
    ), (
        "_anthropic_tool_stream was never registered with _TrackedCancel in "
        "the initial PR; the Anthropic server-tool streaming path was a "
        "no-op for POST /api/inference/cancel"
    )
    assert _finally_calls_tracker_exit(fn)
    assert _has_outer_baseexception_guard(fn)
    assert _loop_checks_cancel_event(fn), (
        "Anthropic tool stream's main loop must check cancel_event.is_set() "
        "-- request.is_disconnected() alone is swallowed by proxies"
    )


def test_anthropic_plain_stream_registers_and_checks_cancel_event():
    fn = _find(_ROUTES_TREE, "_anthropic_plain_stream")
    params = {a.arg for a in fn.args.args + fn.args.kwonlyargs}
    assert {"cancel_id", "session_id"} <= params
    assert _has_tracker_call_with(
        fn, "cancel_id", "session_id", "message_id"
    )
    assert _finally_calls_tracker_exit(fn)
    assert _has_outer_baseexception_guard(fn)
    assert _loop_checks_cancel_event(fn)


# ── The top-level handler threads cancel_id/session_id ───────


def test_anthropic_messages_threads_cancel_id_and_session_id():
    fn = _find(_ROUTES_TREE, "anthropic_messages")
    body_src = ast.unparse(fn)

    # Three distinct call sites must all receive payload.cancel_id.
    assert body_src.count("cancel_id=payload.cancel_id") >= 3, (
        "anthropic_messages must pass payload.cancel_id to all three "
        "streaming helpers (passthrough, server-tool, plain) -- found "
        f"{body_src.count('cancel_id=payload.cancel_id')} occurrence(s)"
    )
    assert body_src.count("session_id=payload.session_id") >= 3, (
        "anthropic_messages must pass payload.session_id to all three "
        "streaming helpers; server-tool and plain previously did not get it"
    )
