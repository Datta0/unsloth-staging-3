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
    out = strip_tool_markup("8��� </binary data> </tool_call>", final=True)
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
        ("answer</function>\n</tool_call>", "answer"),        # run of orphan closes
        ("value</parameter>", "value"),                        # truncated outer close
    ],
)
def test_trailing_orphan_closes_are_stripped_at_final(text, expected):
    assert strip_tool_markup(text, final=True) == expected


# ── the conservative contract must survive ──────────────────────────

def test_wellformed_call_still_stripped():
    text = (
        "Prefix <tool_call>\n<function=web_search>\n<parameter=query>\nx\n"
        "</parameter>\n</function>\n</tool_call> suffix"
    )
    out = strip_tool_markup(text, final=True)
    assert "<tool_call>" not in out and "</tool_call>" not in out
    assert out == "Prefix  suffix"


def test_literal_close_in_mid_prose_survives():
    # A literal </function> mentioned AFTER a real call (not at EOS) is prose, not a
    # leak, and must survive (mirrors the parser's existing conservative behavior).
    text = (
        "<function=web_search><parameter=query>cats</parameter></function>"
        " Done. The tag </function> closes a call."
    )
    assert strip_tool_markup(text, final=True) == "Done. The tag </function> closes a call."


def test_mid_prose_parameter_tag_survives():
    text = "In XML you write <parameter>x</parameter> inside a tag."
    assert strip_tool_markup(text, final=True) == text


def test_streaming_pass_does_not_strip_trailing_orphan():
    # final=False keeps in-progress markup buffered (the trailing-orphan run arm is
    # end-of-turn only), but still scrubs U+FFFD from live display.
    assert strip_tool_markup("Here is the answer.</tool_call>", final=False) == (
        "Here is the answer.</tool_call>"
    )
    assert "�" not in strip_tool_markup("hi � there", final=False)


# ── record_result scrubs tool output on both boundaries ─────────────

def _decision(name="web_search"):
    return ToolCallDecision(
        action="execute", tool_name=name, arguments={"query": "x"},
        tool_call_id="call_0", key=f"{name}:{{}}",
    )


def test_record_result_scrubs_tool_result_for_model_and_display():
    ctrl = ToolLoopController(tools=[{"function": {"name": "web_search"}}])
    dirty = "Title: Page�� body\x00 text�"
    completion = ctrl.record_result(_decision(), dirty)
    # Fed back to the model:
    assert "�" not in completion.model_message()["content"]
    assert "\x00" not in completion.model_message()["content"]
    # Shown in the tool card:
    assert "�" not in completion.tool_end_payload()["result"]
    assert completion.result == "Title: Page body text"


def test_record_result_keeps_clean_result_intact():
    ctrl = ToolLoopController(tools=[{"function": {"name": "web_search"}}])
    clean = "Title: Florida ACA 2026\nSilver premium: $1,900/mo"
    completion = ctrl.record_result(_decision(), clean)
    assert completion.result == clean
