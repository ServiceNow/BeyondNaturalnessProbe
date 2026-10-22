"""Unit tests for the two programmatic prosodic_boundary strategies."""
from __future__ import annotations

from shared.prosodic_strategies import omit_spaces


CANONICAL_DESCRIPTION = "Canonical sentence with natural prosodic boundaries."


class TestOmitSpaces:
    def test_simple_sentence_strips_all_spaces(self):
        result = omit_spaces("Hello world how are you?")
        assert result["versions"][0]["score"] == 0
        assert result["versions"][0]["sentence"] == "Helloworldhowareyou?"

    def test_score_1_is_original_text(self):
        original = "Hello world how are you?"
        result = omit_spaces(original)
        assert result["versions"][1]["score"] == 1
        assert result["versions"][1]["sentence"] == original
        assert result["versions"][1]["error_description"] == CANONICAL_DESCRIPTION

    def test_collapses_tabs_and_newlines(self):
        result = omit_spaces("a\tb\nc d")
        assert result["versions"][0]["sentence"] == "abcd"

    def test_preserves_internal_punctuation(self):
        result = omit_spaces("Wait, what?")
        assert result["versions"][0]["sentence"] == "Wait,what?"


from shared.prosodic_strategies import omit_punctuation


class TestOmitPunctuation:
    def test_strips_internal_commas(self):
        result = omit_punctuation("Wait, what? Really, now.")
        assert result["versions"][0]["sentence"] == "Wait what? Really now."

    def test_preserves_terminal_period(self):
        result = omit_punctuation("Hello, world.")
        assert result["versions"][0]["sentence"] == "Hello world."

    def test_preserves_terminal_question_mark(self):
        result = omit_punctuation("Wait, what?")
        assert result["versions"][0]["sentence"].endswith("?")

    def test_strips_dashes_semicolons_colons_ellipses(self):
        result = omit_punctuation("She said: well — maybe; not now…")
        # all of  : — ;  …  are gone; only the implicit terminal stays
        s = result["versions"][0]["sentence"]
        for ch in [":", "—", ";", "…"]:
            assert ch not in s, f"{ch!r} should have been stripped"

    def test_score_1_is_original_text(self):
        original = "Wait, what? Really, now."
        result = omit_punctuation(original)
        assert result["versions"][1]["sentence"] == original
        assert result["versions"][1]["score"] == 1

    def test_empty_string_is_safe(self):
        result = omit_punctuation("")
        assert result["versions"][0]["sentence"] == ""
        assert result["versions"][1]["sentence"] == ""


import asyncio

from shared.prosodic_strategies import make_prosodic_generator, STRATEGIES


def test_strategies_constant_lists_all_four():
    assert set(STRATEGIES) == {
        "extra_punctuation",
        "omit_spaces", "omit_punctuation", "preset",
    }


def test_make_prosodic_generator_programmatic_omit_spaces():
    gen = make_prosodic_generator(
        client=None,
        deployment="",
        dataset="emergenttts",
        strategy="omit_spaces",
        category="Syntactic Complexity",
    )
    entry = asyncio.run(gen({"id": 7, "text": "Hello world."}))
    assert entry["utterance_id"] == 7
    assert entry["dataset"] == "emergenttts"
    assert entry["category"] == "Syntactic Complexity"
    assert entry["strategy"] == "omit_spaces"
    assert entry["versions"][0]["sentence"] == "Helloworld."
    assert entry["versions"][1]["sentence"] == "Hello world."


def test_make_prosodic_generator_rejects_unknown_strategy():
    import pytest
    with pytest.raises(ValueError, match="Unknown prosodic strategy"):
        make_prosodic_generator(
            client=None, deployment="", dataset="x", strategy="nope",
        )
