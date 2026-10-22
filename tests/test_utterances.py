"""Tests for utterance loading with category filter."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from shared.utterances import load_utterances


@pytest.fixture
def tmp_json(tmp_path: Path) -> Path:
    p = tmp_path / "ds.json"
    p.write_text(json.dumps({
        "utterances": [
            {"id": 1, "text": "A.", "category": "Foo"},
            {"id": 2, "text": "B.", "category": "Bar"},
            {"id": 3, "text": "C.", "category": "Foo"},
            {"id": 4, "text": "D."},  # no category field
        ]
    }))
    return p


def test_load_without_filter_returns_all(tmp_json):
    items = load_utterances(tmp_json)
    assert [u["id"] for u in items] == [1, 2, 3, 4]


def test_load_with_category_filters(tmp_json):
    items = load_utterances(tmp_json, category="Foo")
    assert [u["id"] for u in items] == [1, 3]


def test_load_with_unknown_category_returns_empty(tmp_json):
    items = load_utterances(tmp_json, category="Quux")
    assert items == []
