from __future__ import annotations

import sys
import types
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from commands.ask import (  # noqa: E402
    _build_sources_embed,
    _extract_message_url_citations,
    _extract_web_search_action_sources,
    _format_sources_block,
    _normalize_source_entries,
    _parse_ask_structured_output,
)


def test_normalize_source_entries_filters_invalid_urls_without_dropping_duplicates() -> None:
    raw = [
        {"title": "OpenAI", "url": "https://openai.com#section"},
        {"title": "Duplicate", "url": "https://openai.com"},
        {"title": "No scheme", "url": "openai.com"},
        {"title": "Mail", "url": "mailto:test@example.com"},
        {"title": "", "url": "https://example.com/path"},
        {"title": "None", "url": None},
    ]

    sources = _normalize_source_entries(raw)

    assert sources == [
        {"title": "OpenAI", "url": "https://openai.com"},
        {"title": "Duplicate", "url": "https://openai.com"},
        {"title": "", "url": "https://example.com/path"},
    ]


def test_extract_message_url_citations_supports_object_payloads_without_dedupe() -> None:
    outputs = [
        types.SimpleNamespace(
            type="message",
            content=[
                types.SimpleNamespace(
                    annotations=[
                        types.SimpleNamespace(
                            type="url_citation",
                            url="https://example.com/a",
                            title="Example A",
                        ),
                        types.SimpleNamespace(
                            type="url_citation",
                            url="https://example.com/a#frag",
                            title="Example A duplicate",
                        ),
                    ]
                )
            ],
        )
    ]

    sources = _extract_message_url_citations(outputs)

    assert sources == [
        {"title": "Example A", "url": "https://example.com/a"},
        {"title": "Example A duplicate", "url": "https://example.com/a"},
    ]


def test_extract_web_search_action_sources_uses_url_only() -> None:
    outputs = [
        {
            "type": "web_search_call",
            "action": {
                "sources": [
                    {"title": "One", "url": "https://one.example/path"},
                ]
            },
        },
        types.SimpleNamespace(
            type="web_search_call",
            action=types.SimpleNamespace(
                sources=[
                    types.SimpleNamespace(title="Two", url="https://two.example/path#frag"),
                    types.SimpleNamespace(title="Bad", url="javascript:alert(1)"),
                ]
            ),
        ),
    ]

    sources = _extract_web_search_action_sources(outputs)

    assert sources == [
        {"title": "", "url": "https://one.example/path"},
        {"title": "", "url": "https://two.example/path"},
    ]


def test_format_sources_block_supports_main_and_sauce_sections() -> None:
    text = _format_sources_block(
        main_sources=[{"title": "OpenAI", "url": "https://openai.com"}],
        sauce_sources=[{"title": "", "url": "https://youtube.com"}],
    )

    assert text == "🧭Main sauce\n[1] https://openai.com\n\n🔗Sources\n[1] https://youtube.com"


def test_format_sources_block_empty_when_no_sources() -> None:
    assert _format_sources_block(main_sources=[], sauce_sources=[]) == ""


def test_build_sources_embed_returns_discord_embed_with_sources() -> None:
    embed = _build_sources_embed(
        main_sources=[{"title": "OpenAI", "url": "https://openai.com"}],
        sauce_sources=[{"title": "", "url": "https://youtube.com"}],
    )

    assert embed is not None
    assert embed.title == "🔎 Source links"
    assert embed.description == "🧭Main sauce\n[1] https://openai.com\n\n🔗Sources\n[1] https://youtube.com"


def test_build_sources_embed_returns_none_when_sources_are_empty() -> None:
    assert _build_sources_embed(main_sources=[], sauce_sources=[]) is None


def test_parse_structured_output_accepts_sources_field() -> None:
    raw = """
    {
      "title": "t",
      "answer": "a",
      "reasoning_summary": [],
      "tool_timeline": [],
      "artifacts": [],
      "sources": [
        {"title": "OpenAI", "url": "https://openai.com"}
      ]
    }
    """

    parsed = _parse_ask_structured_output(raw)

    assert parsed is not None
    assert parsed["sources"] == [{"title": "OpenAI", "url": "https://openai.com"}]


def test_parse_structured_output_rejects_missing_sources_field() -> None:
    raw = """
    {
      "title": "t",
      "answer": "a",
      "reasoning_summary": [],
      "tool_timeline": [],
      "artifacts": []
    }
    """

    parsed = _parse_ask_structured_output(raw)

    assert parsed is None
