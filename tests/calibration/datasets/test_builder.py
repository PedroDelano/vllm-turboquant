"""Builder pipeline tests.

Uses MockTokenizer + StubAdapter so nothing here depends on real models
or network. Tests exercise: JSONL output shape, min/max-token filtering,
num-prompts stop condition, tool-role fallback, tools-fallback modes,
empty-output error, missing-template error.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from calibration.datasets.base import Conversation
from calibration.datasets.builder import BuildReport, build

from .conftest import MockTokenizer, StubAdapter


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_build_writes_dual_key_jsonl(
    tmp_path, mock_tokenizer, simple_conversations
):
    output = tmp_path / "out.jsonl"
    adapter = StubAdapter(simple_conversations)

    report = build(
        adapter,
        tokenizer=mock_tokenizer,
        output_path=output,
        num_prompts=10,
        min_tokens=1,
        max_tokens=128,
    )

    assert isinstance(report, BuildReport)
    assert report.kept == 1
    assert report.skipped == 0
    assert report.output_path == output

    rows = _read_jsonl(output)
    assert len(rows) == 1
    assert set(rows[0].keys()) == {"prompt", "text"}
    assert rows[0]["prompt"] == rows[0]["text"]
    assert "<user>" in rows[0]["text"]
    assert "what" in rows[0]["text"]


def test_build_drops_short_prompts(tmp_path, mock_tokenizer):
    short = Conversation(messages=[{"role": "user", "content": "hi"}])
    long = Conversation(
        messages=[
            {"role": "user", "content": " ".join(["word"] * 200)},
        ]
    )
    adapter = StubAdapter([short, long])
    output = tmp_path / "out.jsonl"

    report = build(
        adapter,
        tokenizer=mock_tokenizer,
        output_path=output,
        num_prompts=10,
        min_tokens=50,
        max_tokens=1024,
    )

    assert report.kept == 1
    assert report.skipped == 1


def test_build_truncates_at_max_tokens(tmp_path, mock_tokenizer):
    long = Conversation(
        messages=[
            {"role": "user", "content": " ".join(["word"] * 500)},
        ]
    )
    adapter = StubAdapter([long])
    output = tmp_path / "out.jsonl"

    build(
        adapter,
        tokenizer=mock_tokenizer,
        output_path=output,
        num_prompts=1,
        min_tokens=1,
        max_tokens=64,
    )

    rows = _read_jsonl(output)
    assert len(rows) == 1
    # MockTokenizer.decode returns the first len(ids) words of the last text.
    assert len(rows[0]["text"].split()) == 64


def test_build_stops_at_num_prompts(tmp_path, mock_tokenizer):
    convs = [
        Conversation(
            messages=[{"role": "user", "content": "hello world foo bar baz"}]
        )
        for _ in range(10)
    ]
    adapter = StubAdapter(convs)
    output = tmp_path / "out.jsonl"

    report = build(
        adapter,
        tokenizer=mock_tokenizer,
        output_path=output,
        num_prompts=3,
        min_tokens=1,
        max_tokens=1024,
    )

    assert report.kept == 3
    rows = _read_jsonl(output)
    assert len(rows) == 3


def test_build_rewrites_tool_role_when_template_rejects_it(
    tmp_path, mock_tokenizer_rejects_tool_role
):
    conv = Conversation(
        messages=[
            {"role": "user", "content": "get me the weather"},
            {"role": "assistant", "content": "calling tool"},
            {"role": "tool", "content": "sunny"},
            {"role": "assistant", "content": "it is sunny"},
        ]
    )
    adapter = StubAdapter([conv])
    output = tmp_path / "out.jsonl"

    build(
        adapter,
        tokenizer=mock_tokenizer_rejects_tool_role,
        output_path=output,
        num_prompts=1,
        min_tokens=1,
        max_tokens=1024,
    )

    rows = _read_jsonl(output)
    assert len(rows) == 1
    assert "<tool>" not in rows[0]["text"]
    assert "[tool response]" in rows[0]["text"]


def test_tools_fallback_error_raises(
    tmp_path, mock_tokenizer_rejects_tools, conversations_with_tools
):
    adapter = StubAdapter(conversations_with_tools)
    output = tmp_path / "out.jsonl"
    with pytest.raises(ValueError, match="template does not accept tools"):
        build(
            adapter,
            tokenizer=mock_tokenizer_rejects_tools,
            output_path=output,
            num_prompts=1,
            min_tokens=1,
            max_tokens=1024,
            tools_fallback="error",
        )


def test_tools_fallback_drop_skips(
    tmp_path, mock_tokenizer_rejects_tools, conversations_with_tools
):
    convs = conversations_with_tools + [
        Conversation(
            messages=[{"role": "user", "content": "hello world foo bar"}],
            tools=None,
        )
    ]
    adapter = StubAdapter(convs)
    output = tmp_path / "out.jsonl"

    report = build(
        adapter,
        tokenizer=mock_tokenizer_rejects_tools,
        output_path=output,
        num_prompts=10,
        min_tokens=1,
        max_tokens=1024,
        tools_fallback="drop",
    )
    assert report.kept == 1
    assert report.skipped == 1


def test_tools_fallback_render_as_system_embeds_tools_json(
    tmp_path, mock_tokenizer_rejects_tools, conversations_with_tools
):
    adapter = StubAdapter(conversations_with_tools)
    output = tmp_path / "out.jsonl"

    build(
        adapter,
        tokenizer=mock_tokenizer_rejects_tools,
        output_path=output,
        num_prompts=1,
        min_tokens=1,
        max_tokens=1024,
        tools_fallback="render-as-system",
    )
    rows = _read_jsonl(output)
    assert len(rows) == 1
    # Rendered-as-system puts tool JSON inside a system message; the
    # MockTokenizer's apply_chat_template wraps system content in <system>.
    assert "<system>" in rows[0]["text"]
    assert "get_weather" in rows[0]["text"]


def test_build_rejects_tokenizer_without_chat_template(
    tmp_path, mock_tokenizer_no_template, simple_conversations
):
    adapter = StubAdapter(simple_conversations)
    output = tmp_path / "out.jsonl"
    with pytest.raises(SystemExit, match="no chat_template"):
        build(
            adapter,
            tokenizer=mock_tokenizer_no_template,
            output_path=output,
            num_prompts=1,
            min_tokens=1,
            max_tokens=1024,
        )


def test_build_raises_when_all_rows_filtered(tmp_path, mock_tokenizer):
    convs = [
        Conversation(messages=[{"role": "user", "content": "hi"}])
        for _ in range(5)
    ]
    adapter = StubAdapter(convs)
    output = tmp_path / "out.jsonl"
    with pytest.raises(RuntimeError, match="kept 0 prompts"):
        build(
            adapter,
            tokenizer=mock_tokenizer,
            output_path=output,
            num_prompts=3,
            min_tokens=50,
            max_tokens=1024,
        )


def test_build_respects_explicit_split_override(
    tmp_path, mock_tokenizer, simple_conversations
):
    seen: list[str] = []

    class RecordingAdapter:
        name = "rec"
        dataset_id = "rec/rec"
        default_split = "train"

        def iter_conversations(self, *, split):
            seen.append(split)
            yield from simple_conversations

    build(
        RecordingAdapter(),
        tokenizer=mock_tokenizer,
        output_path=tmp_path / "out.jsonl",
        num_prompts=1,
        min_tokens=1,
        max_tokens=1024,
        split="validation",
    )
    assert seen == ["validation"]
