# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Regression tests for the MTP GGUF tool-call garbage bug (issue #7084).

On an MTP GGUF model (e.g. Qwen3.6-27B-MTP), speculative decoding on a quantized
target (ggml-org/llama.cpp#25618) intermittently emits byte-fallback garbage that
llama-server forwards as U+FFFD plus an orphaned ``</tool_call>`` close whose
opener was drained or ``�``-mangled. The parser-side display strip
(``strip_tool_markup``, used for live GGUF streaming display) had no arm for a bare
orphan close, so the reporter saw ``8��� </binary data> </tool_call>`` in chat.
These tests pin the two boundary defenses:

  1. ``strip_tool_markup`` scrubs U+FFFD / control chars and removes a trailing run
     of orphan closes, while keeping well-formed stripping and mid-prose literals.
  2. ``sanitize_control_chars`` drops the garbage but keeps ``\t \n \r`` / ESC.
  3. ``ToolLoopController.record_result`` scrubs the same garbage from a tool
     result before it reaches the model or the tool card.
"""

import pytest

from core.inference.tool_call_parser import sanitize_control_chars, strip_tool_markup
from core.inference.tool_loop_controller import ToolCallDecision, ToolLoopController


# ── sanitize_control_chars ──────────────────────────────────────────


def test_sanitize_drops_replacement_and_control_chars():
    assert sanitize_control_chars("8��� ok") == "8 ok"
    assert sanitize_control_chars("a\x00b\x7fc\x9fd") == "abcd"


def test_sanitize_keeps_tab_newline_cr_and_esc():
    # ESC (\x1b) is preserved so terminal ANSI in a tool result survives.
    assert sanitize_control_chars("a\tb\nc\r\n\x1b[0m") == "a\tb\nc\r\n\x1b[0m"


def test_sanitize_noop_on_clean_text():
    s = "Perfectly normal answer with a supplementary-plane char 𠀀 and kanji 美味しい."
    assert sanitize_control_chars(s) == s


# ── strip_tool_markup: the reporter's exact garbage ────────────────


def test_reporter_garbage_is_cleaned():
    # The exact string from issue #7084 screenshot 2. Before the fix this passed
    # through unchanged (both the `�` and the orphan `</tool_call>` leaked).
    out = strip_tool_markup("8��� </binary data> </tool_call>", final = True)
    assert "�" not in out
    assert "</tool_call>" not in out
    # `</binary data>` is the model's own hallucinated text, not a Studio token, so
    # it is intentionally left as-is (Studio never invents/strips arbitrary prose).
    assert out == "8 </binary data>"


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Here is the answer.</tool_call>", "Here is the answer."),
        ("Done.</function>", "Done."),
        ("x<tool_call|>", "x"),
        ("answer</function>\n</tool_call>", "answer"),  # run of orphan closes
        ("value</parameter>", "value"),  # truncated outer close
        ("Here is the answer.</tool_call>\n", "Here is the answer."),  # trailing newline
        ("Done.</function>  ", "Done."),  # trailing spaces
        ("8��� </tool_call>\n", "8"),  # reporter's garbage + trailing newline
    ],
)
def test_trailing_orphan_closes_are_stripped_at_final(text, expected):
    assert strip_tool_markup(text, final = True) == expected


# ── the conservative contract must survive ──────────────────────────


def test_streamed_tool_output_is_sanitized():
    # A live tool chunk carrying U+FFFD must be scrubbed before it reaches the UI,
    # since record_result only cleans the final result and the UI keeps the stream.
    from core.inference.tool_stream_exec import stream_tool_execution

    def invoke(on_output):
        on_output("ok\t█� chunk")
        return "ok chunk"

    texts = [
        ev["text"]
        for ev in stream_tool_execution(invoke, tool_name = "python")
        if ev.get("type") == "tool_output"
    ]
    assert texts and all("�" not in t for t in texts)
    assert "\t" in "".join(texts)  # legitimate whitespace preserved


def test_wellformed_call_still_stripped():
    text = (
        "Prefix <tool_call>\n<function=web_search>\n<parameter=query>\nx\n"
        "</parameter>\n</function>\n</tool_call> suffix"
    )
    out = strip_tool_markup(text, final = True)
    assert "<tool_call>" not in out and "</tool_call>" not in out
    assert out == "Prefix  suffix"


def test_literal_close_in_mid_prose_survives():
    # A literal </function> mentioned AFTER a real call (not at EOS) is prose, not a
    # leak, and must survive (mirrors the parser's existing conservative behavior).
    text = (
        "<function=web_search><parameter=query>cats</parameter></function>"
        " Done. The tag </function> closes a call."
    )
    assert strip_tool_markup(text, final = True) == "Done. The tag </function> closes a call."


def test_mid_prose_parameter_tag_survives():
    text = "In XML you write <parameter>x</parameter> inside a tag."
    assert strip_tool_markup(text, final = True) == text


def test_streaming_pass_does_not_strip_trailing_orphan():
    # final=False keeps in-progress markup buffered (the trailing-orphan run arm is
    # end-of-turn only), but still scrubs U+FFFD from live display.
    assert strip_tool_markup("Here is the answer.</tool_call>", final = False) == (
        "Here is the answer.</tool_call>"
    )
    assert "�" not in strip_tool_markup("hi � there", final = False)


# ── record_result scrubs tool output on both boundaries ─────────────


def _decision(name = "web_search"):
    return ToolCallDecision(
        action = "execute",
        tool_name = name,
        arguments = {"query": "x"},
        tool_call_id = "call_0",
        key = f"{name}:{{}}",
    )


def test_record_result_scrubs_tool_result_for_model_and_display():
    ctrl = ToolLoopController(tools = [{"function": {"name": "web_search"}}])
    dirty = "Title: Page�� body\x00 text�"
    completion = ctrl.record_result(_decision(), dirty)
    # Fed back to the model:
    assert "�" not in completion.model_message()["content"]
    assert "\x00" not in completion.model_message()["content"]
    # Shown in the tool card:
    assert "�" not in completion.tool_end_payload()["result"]
    assert completion.result == "Title: Page body text"


def test_record_result_keeps_clean_result_intact():
    ctrl = ToolLoopController(tools = [{"function": {"name": "web_search"}}])
    clean = "Title: Florida ACA 2026\nSilver premium: $1,900/mo"
    completion = ctrl.record_result(_decision(), clean)
    assert completion.result == clean


# ── final synthetic-answer pass content sanitization ───────────────


def _sse(delta: dict) -> str:
    import json
    return "data: " + json.dumps({"choices": [{"index": 0, "delta": delta}]}) + "\n"


def _done() -> str:
    return "data: [DONE]\n"


def test_final_pass_content_is_sanitized_without_auto_heal(monkeypatch):
    """The tool-cap final answer pass must scrub U+FFFD / control chars itself.

    With ``auto_heal_tool_calls=False`` the display strip is a no-op, so on an
    MTP GGUF model the final chat response would otherwise leak the byte-fallback
    garbage every other content channel already scrubs (#7084 / PR #7243).
    """
    import contextlib
    import json as _json

    from core.inference.llama_cpp import LlamaCppBackend

    backend = LlamaCppBackend.__new__(LlamaCppBackend)
    backend._process = object()
    backend._healthy = True
    backend._port = 48847
    backend._api_key = None
    backend._effective_context_length = 4096
    backend._supports_reasoning = False
    backend._reasoning_always_on = False
    backend._reasoning_style = "enable_thinking"
    backend._supports_preserve_thinking = False

    tool_stream = [
        _sse(
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "render_html",
                            "arguments": _json.dumps({"code": "<html>ok</html>"}),
                        },
                    }
                ]
            }
        ),
        _done(),
    ]
    # A content token carrying MTP byte-fallback garbage in the final pass.
    final_stream = [_sse({"content": "Final � answer\x00 here."}), _done()]
    streams = [tool_stream, final_stream]

    @contextlib.contextmanager
    def fake_stream_with_retry(
        _client,
        _url,
        _payload,
        _cancel_event,
        headers = None,
        first_token_deadline = None,
    ):
        yield type("FakeResponse", (), {"status_code": 200, "chunks": streams.pop(0)})()

    def fake_iter_text_cancellable(
        response,
        _cancel_event,
        first_token_deadline = None,
    ):
        yield from response.chunks

    monkeypatch.setattr(backend, "_stream_with_retry", fake_stream_with_retry)
    monkeypatch.setattr(backend, "_iter_text_cancellable", fake_iter_text_cancellable)
    monkeypatch.setattr(
        "core.inference.tools.execute_tool",
        lambda name, arguments, **_kwargs: "Rendered HTML canvas: Done.",
    )

    events = list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "render then answer"}],
            tools = [{"type": "function", "function": {"name": "render_html"}}],
            max_tool_iterations = 1,
            auto_heal_tool_calls = False,
        )
    )

    final_content = [
        e for e in events if e.get("type") == "content" and "Final" in e.get("text", "")
    ]
    assert final_content, "final-pass content event missing"
    text = final_content[-1]["text"]
    assert "�" not in text
    assert "\x00" not in text
    # Clean text survives byte-for-byte (only the garbage is dropped).
    assert "answer" in text and "here." in text
