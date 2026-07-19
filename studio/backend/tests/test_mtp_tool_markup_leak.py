# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Regression tests for the MTP GGUF tool-call garbage bug (issue #7084).

On an MTP GGUF model, speculative decoding on a quantized target
(ggml-org/llama.cpp#25618) emits byte-fallback garbage that llama-server forwards as
U+FFFD plus an orphaned ``</tool_call>`` whose opener was drained/``�``-mangled;
``strip_tool_markup`` had no arm for a bare orphan close, so the reporter saw
``8��� </binary data> </tool_call>`` in chat. Pins the boundary defenses:

  1. ``strip_tool_markup`` scrubs U+FFFD and removes a trailing orphan-close run,
     while keeping well-formed stripping and mid-prose literals.
  2. ``sanitize_control_chars`` drops garbage but keeps ``\t \n \r`` / ESC.
  3. ``ToolLoopController.record_result`` scrubs a tool result before the model/card.
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
    # Exact string from issue #7084; before the fix both `�` and the orphan `</tool_call>` leaked.
    out = strip_tool_markup("8��� </binary data> </tool_call>", final = True)
    assert "�" not in out
    assert "</tool_call>" not in out
    # `</binary data>` is model-hallucinated prose, not a Studio token, so it is left as-is.
    assert out == "8 </binary data>"


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Here is the answer.</tool_call>", "Here is the answer."),
        ("x<tool_call|>", "x"),
        ("answer</function>\n</tool_call>", "answer"),  # nested leak run ending in </tool_call>
        ("Here is the answer.</tool_call>\n", "Here is the answer."),  # trailing newline
        ("8��� </tool_call>\n", "8"),  # reporter's garbage + trailing newline
    ],
)
def test_trailing_orphan_closes_are_stripped_at_final(text, expected):
    assert strip_tool_markup(text, final = True) == expected


@pytest.mark.parametrize(
    "text",
    [
        "Done.</function>",  # lone close, no </tool_call> sentinel
        "value</parameter>",
        "The XML closing tag is </function>",  # a code/XML answer ending on a literal
        "In XML the outer tag closes with </parameter>",
    ],
)
def test_trailing_literal_close_without_tool_call_survives(text):
    # A trailing </function> / </parameter> with no </tool_call> sentinel reads as a
    # code/XML literal, not a leak, so it survives (a real leak carries </tool_call>).
    assert strip_tool_markup(text, final = True) == text


def test_long_trailing_orphan_run_is_fully_stripped():
    # A pathological orphan-close run is stripped whole; the </tool_call> sentinel gates removal.
    payload = "answer" + (" </tool_call>" * 500)
    assert strip_tool_markup(payload, final = True) == "answer"


def test_trailing_orphan_strip_is_linear_not_redos():
    # Linear scan, not a backtracking regex: a long close run ending in a near-miss token
    # used to backtrack catastrophically (~4s). The helper returns well under a second and
    # leaves the near-miss tail untouched.
    import time

    payload = (" </tool_call>" * 2000) + " </tool_calX>"
    start = time.perf_counter()
    out = strip_tool_markup(payload, final = True)
    elapsed = time.perf_counter() - start
    assert elapsed < 1.0, f"strip took {elapsed:.3f}s (possible ReDoS regression)"
    assert out.endswith("</tool_calX>")


# ── the conservative contract must survive ──────────────────────────


def test_streamed_tool_output_is_sanitized():
    # A live chunk's U+FFFD must be scrubbed before the UI: record_result only cleans
    # the final result and the UI keeps the stream.
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
    # A literal </function> after a real call (not at EOS) is prose, not a leak, and survives.
    text = (
        "<function=web_search><parameter=query>cats</parameter></function>"
        " Done. The tag </function> closes a call."
    )
    assert strip_tool_markup(text, final = True) == "Done. The tag </function> closes a call."


def test_mid_prose_parameter_tag_survives():
    text = "In XML you write <parameter>x</parameter> inside a tag."
    assert strip_tool_markup(text, final = True) == text


def test_streaming_pass_does_not_strip_trailing_orphan():
    # final=False keeps in-progress markup buffered (orphan-run arm is end-of-turn only)
    # but still scrubs U+FFFD.
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
    """The tool-cap final answer pass must scrub U+FFFD itself: with
    ``auto_heal_tool_calls=False`` the display strip is a no-op, so an MTP GGUF final
    response would otherwise leak byte-fallback garbage (#7084 / PR #7243)."""
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


# ── the GGUF streaming _seg must scrub the orphan close, not only the parser ──
#
# The bug (Codex 3610967410): the llama.cpp GGUF streaming path has its own duplicate
# stripper ``_strip_tool_markup_streaming._seg`` that mirrors the parser's ``seg_final``
# block (Gemma / function-XML / GLM scans + _TOOL_ALL_PATS then _REHEARSAL_TAIL_STRIP_RE)
# but never called ``_strip_trailing_orphan_close_run``. So a GGUF-streamed plain answer
# whose ``<tool_call>`` opener was drained / U+FFFD-mangled (MTP byte-fallback) streamed its
# trailing orphan ``</tool_call>`` into the bubble, while the safetensors path scrubbed it
# via ``strip_tool_markup(final=True)``. ``_seg`` runs both mid-stream and at end-of-stream,
# so the fix must live in ``_seg`` (not only the end-of-stream call site).


class TestMTPStreamingOrphanCloseLeak:
    """Regression for the GGUF streaming ``_seg`` orphan-close leak (Codex 3610967410)."""

    @staticmethod
    def _seg_final(seg: str, *, orphan_strip: bool) -> str:
        """Faithful reconstruction of the GGUF streaming ``_seg`` on the last segment
        (``is_last=True``), name-agnostic, built from the same shared helpers the closure
        imports. ``orphan_strip=False`` reproduces the pre-fix pipeline; ``orphan_strip=True``
        inserts the fix in the parser's order (orphan-strip then rehearsal-tail)."""
        from core.inference.tool_call_parser import (
            _TOOL_ALL_PATS,
            _strip_function_xml_calls,
            _strip_gemma_wrapperless_calls,
            _strip_glm_calls,
            _strip_mistral_closed_calls,
            _strip_trailing_orphan_close_run,
        )
        from core.tool_healing import (
            _REHEARSAL_TAIL_STRIP_RE,
            _strip_bracket_tag_calls,
            apply_tool_strip_patterns,
        )

        seg = _strip_mistral_closed_calls(seg)
        seg = _strip_bracket_tag_calls(seg, enabled_tool_names = None)
        seg = _strip_gemma_wrapperless_calls(seg, None)
        seg = _strip_function_xml_calls(seg, final = True)
        seg = _strip_glm_calls(seg, final = True)
        for pat in _TOOL_ALL_PATS:
            seg = pat.sub("", seg)
        if orphan_strip:
            seg = _strip_trailing_orphan_close_run(seg)
        seg = apply_tool_strip_patterns(seg, [_REHEARSAL_TAIL_STRIP_RE], enabled_tool_names = None)
        return seg

    def test_orphan_close_leaks_pre_fix_and_is_scrubbed_post_fix(self):
        # Pre-fix (no orphan strip) the streaming final segment leaks the close; the fix
        # scrubs it and matches the shared parser-final output.
        assert self._seg_final("answer</tool_call>", orphan_strip = False) == ("answer</tool_call>")
        assert self._seg_final("answer</tool_call>", orphan_strip = True) == "answer"
        assert strip_tool_markup("answer</tool_call>", final = True) == "answer"

    def test_reporter_byte_fallback_case(self):
        # MTP byte-fallback: token ingestion in the GGUF loop sanitizes control chars before
        # cumulative_display reaches _seg, so the stripper sees a bare orphan close.
        cleaned = sanitize_control_chars("answer���</tool_call>")
        assert cleaned == "answer</tool_call>"
        assert self._seg_final(cleaned, orphan_strip = False) == "answer</tool_call>"
        assert self._seg_final(cleaned, orphan_strip = True) == "answer"
        # The intervening-whitespace variant: the close (and the run's whitespace) is scrubbed;
        # the streaming _seg leaves a cosmetic trailing space (no final trim), but the leaking
        # </tool_call> token is gone.
        spaced = sanitize_control_chars("answer��� </tool_call>")
        assert spaced == "answer </tool_call>"
        assert "</tool_call>" not in self._seg_final(spaced, orphan_strip = True)

    def test_no_over_strip_on_plain_message(self):
        # A legitimate message with no orphan-close sentinel is untouched, and the scrub is
        # a no-op relative to the pre-fix pipeline.
        for text in ("hello world, no markup here", "The XML closing tag is </function>"):
            assert self._seg_final(text, orphan_strip = True) == text
            assert self._seg_final(text, orphan_strip = True) == self._seg_final(
                text, orphan_strip = False
            )

    def test_genuine_tool_call_not_double_stripped(self):
        # A real closed call is fully removed, and the extra scrub does not corrupt it.
        call = '<tool_call>{"name":"web_search","arguments":{"query":"x"}}</tool_call>'
        assert self._seg_final(call, orphan_strip = True) == ""
        assert self._seg_final(call, orphan_strip = True) == self._seg_final(call, orphan_strip = False)

    def test_streaming_seg_matches_shared_parser_final(self):
        for text in (
            "answer</tool_call>",
            "answer</tool_call> </tool_call>",
            "hello world",
            '<tool_call>{"name":"web_search","arguments":{"query":"x"}}</tool_call>',
        ):
            assert self._seg_final(text, orphan_strip = True) == strip_tool_markup(text, final = True)

    def test_real_streaming_closure_wires_in_the_scrub(self):
        # Pin the fix into the actual code: the GGUF streaming stripper's final-segment
        # block must call _strip_trailing_orphan_close_run (not only the shared final path).
        import inspect

        from core.inference.llama_cpp import LlamaCppBackend

        src = inspect.getsource(LlamaCppBackend.generate_chat_completion_with_tools)
        assert "_strip_trailing_orphan_close_run(seg)" in src, (
            "GGUF _strip_tool_markup_streaming must scrub trailing orphan closes like "
            "strip_tool_markup(final=True)"
        )


# ── Kimi + DeepSeek end-of-turn closers belong to the orphan-close set ──
#
# The bug (Codex 3611159411): ``_ORPHAN_CLOSE_TOKENS`` / ``_ORPHAN_SENTINELS`` only listed
# the Qwen/Gemma/function-XML closers, so a Kimi ``<|tool_call_end|><|tool_calls_section_end|>``
# or the DeepSeek ``<｜tool▁call▁end｜><｜tool▁calls▁end｜>`` run whose opener was drained /
# U+FFFD-mangled leaked verbatim. These are back-to-back special tokens (a contiguous
# end-of-text run the linear scanner handles) and are never legit prose, so they join the
# sentinel set like ``<tool_call|>``.


class TestKimiDeepSeekOrphanCloses:
    def _tokens(self):
        from core.inference.tool_call_parser import (
            _DEEPSEEK_CALL_END,
            _DEEPSEEK_END,
            _KIMI_CALL_END,
            _KIMI_SECTION_END,
        )
        return _KIMI_CALL_END, _KIMI_SECTION_END, _DEEPSEEK_CALL_END, _DEEPSEEK_END

    def test_tokens_registered_in_orphan_sets(self):
        from core.inference.tool_call_parser import (
            _ORPHAN_CLOSE_TOKENS,
            _ORPHAN_SENTINELS,
        )
        for tok in self._tokens():
            assert tok in _ORPHAN_CLOSE_TOKENS
            assert tok in _ORPHAN_SENTINELS

    def test_kimi_trailing_closers_stripped_at_final(self):
        kimi_end, kimi_section_end, *_ = self._tokens()
        text = "answer " + kimi_end + kimi_section_end
        assert strip_tool_markup(text, final = True) == "answer"

    def test_deepseek_trailing_closers_stripped_at_final(self):
        _, _, ds_call_end, ds_end = self._tokens()
        text = "answer " + ds_call_end + ds_end
        assert strip_tool_markup(text, final = True) == "answer"

    def test_streaming_stripper_scrubs_kimi_and_deepseek(self):
        # The shared ``_strip_trailing_orphan_close_run`` is what both stream paths (GGUF _seg
        # and safetensors _seg) call, so exercising it pins the streaming behavior too.
        from core.inference.tool_call_parser import _strip_trailing_orphan_close_run

        kimi_end, kimi_section_end, ds_call_end, ds_end = self._tokens()
        assert (
            _strip_trailing_orphan_close_run("answer " + kimi_end + kimi_section_end) == "answer "
        )
        assert _strip_trailing_orphan_close_run("answer " + ds_call_end + ds_end) == "answer "
        # And through the safetensors streaming stripper (which threads the same helper).
        from core.inference.safetensors_agentic import strip_tool_markup_streaming

        assert strip_tool_markup_streaming("answer" + kimi_end + kimi_section_end) == "answer"
        assert strip_tool_markup_streaming("answer" + ds_call_end + ds_end) == "answer"

    def test_single_trailing_closer_also_stripped(self):
        # Each is a sentinel on its own, so even a lone trailing closer is scrubbed.
        kimi_end, kimi_section_end, ds_call_end, ds_end = self._tokens()
        for tok in (kimi_end, kimi_section_end, ds_call_end, ds_end):
            assert strip_tool_markup("answer " + tok, final = True) == "answer"

    def test_literal_token_in_mid_prose_survives(self):
        # Only TRAILING orphans are stripped: a token embedded in prose (with real text after)
        # is not a trailing run, so it is kept and a plain answer is never over-stripped.
        kimi_end, *_ = self._tokens()
        text = "the token " + kimi_end + " appears mid sentence"
        assert strip_tool_markup(text, final = True) == text

    def test_genuine_kimi_tool_call_still_parsed(self):
        # A well-formed Kimi call still parses (the orphan-close addition does not disturb the
        # structured parse path).
        from core.inference.tool_call_parser import parse_tool_calls_from_text

        (
            kimi_end,
            kimi_section_end,
            *_,
        ) = self._tokens()
        from core.inference.tool_call_parser import (
            _KIMI_ARG_BEGIN,
            _KIMI_CALL_BEGIN,
            _KIMI_SECTION_BEGIN,
        )

        call = (
            _KIMI_SECTION_BEGIN
            + _KIMI_CALL_BEGIN
            + "functions.web_search:0"
            + _KIMI_ARG_BEGIN
            + '{"query":"x"}'
            + kimi_end
            + kimi_section_end
        )
        result = parse_tool_calls_from_text(call)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "web_search"
        assert "x" in result[0]["function"]["arguments"]
