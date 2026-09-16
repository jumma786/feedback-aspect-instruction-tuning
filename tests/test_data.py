"""Tests for the regrouping and prompt-building logic.

These run on hand-built rows rather than the real download, so CI does not
depend on a third-party dataset staying available. The shape of the fake rows
matches the real ones exactly: one row per (review, aspect) pair.
"""

from __future__ import annotations

import json

import pytest

from src.data import Example, _group, build_messages, describe


def _row(text, span, label, ordinal=0):
    return {"text": text, "span": span, "label": label, "ordinal": ordinal}


class TestGrouping:
    def test_one_row_per_aspect_becomes_one_example_per_review(self):
        rows = [
            _row("The staff were rude but the food was great.", "staff", "negative"),
            _row("The staff were rude but the food was great.", "food", "positive"),
        ]
        grouped = _group(rows)
        assert len(grouped) == 1
        assert grouped[0].aspects == (("staff", "negative"), ("food", "positive"))

    def test_annotation_order_is_preserved(self):
        rows = [
            _row("a b c", "c", "neutral"),
            _row("a b c", "a", "positive"),
            _row("a b c", "b", "negative"),
        ]
        assert _group(rows)[0].aspects == (
            ("c", "neutral"),
            ("a", "positive"),
            ("b", "negative"),
        )

    def test_distinct_reviews_stay_separate(self):
        rows = [_row("first review", "x", "positive"), _row("second review", "y", "negative")]
        assert len(_group(rows)) == 2

    def test_unlabelled_rows_are_dropped(self):
        """The real mirror ships an unlabelled test split; it must not sneak in."""
        rows = [_row("some review", "thing", ""), _row("some review", "other", "positive")]
        grouped = _group(rows)
        assert len(grouped) == 1
        assert grouped[0].aspects == (("other", "positive"),)

    def test_a_fully_unlabelled_split_groups_to_nothing(self):
        rows = [_row("a", "x", ""), _row("b", "y", "")]
        assert _group(rows) == []

    def test_unknown_sentiment_labels_are_dropped(self):
        rows = [_row("review", "thing", "delighted")]
        assert _group(rows) == []

    def test_conflict_label_is_kept(self):
        """Dropping 'conflict' would quietly inflate sentiment accuracy."""
        rows = [_row("review", "thing", "conflict")]
        assert _group(rows)[0].aspects == (("thing", "conflict"),)

    def test_duplicate_annotations_are_collapsed(self):
        rows = [
            _row("review", "thing", "positive"),
            _row("review", "thing", "positive", ordinal=1),
        ]
        assert _group(rows)[0].aspects == (("thing", "positive"),)

    def test_blank_text_or_span_is_dropped(self):
        rows = [_row("", "x", "positive"), _row("review", "", "positive")]
        assert _group(rows) == []

    def test_labels_are_case_normalised(self):
        rows = [_row("review", "thing", "POSITIVE")]
        assert _group(rows)[0].aspects == (("thing", "positive"),)


class TestTargets:
    def test_target_json_is_valid_and_ordered(self):
        example = Example("review", (("staff", "negative"), ("food", "positive")))
        payload = json.loads(example.target_json())
        assert payload == {
            "aspects": [
                {"term": "staff", "sentiment": "negative"},
                {"term": "food", "sentiment": "positive"},
            ]
        }

    def test_empty_aspects_render_as_empty_list(self):
        assert json.loads(Example("review", ()).target_json()) == {"aspects": []}

    def test_non_ascii_text_survives(self):
        example = Example("café was lovely", (("café", "positive"),))
        assert "café" in example.target_json()


class TestPrompts:
    def test_inference_prompt_has_no_answer(self):
        messages = build_messages(Example("review", (("x", "positive"),)), include_answer=False)
        assert [m["role"] for m in messages] == ["system", "user"]

    def test_training_prompt_appends_gold_answer(self):
        example = Example("review", (("x", "positive"),))
        messages = build_messages(example, include_answer=True)
        assert [m["role"] for m in messages] == ["system", "user", "assistant"]
        assert json.loads(messages[-1]["content"]) == json.loads(example.target_json())

    def test_review_text_reaches_the_user_turn(self):
        messages = build_messages(Example("the bread is great", ()), include_answer=False)
        assert "the bread is great" in messages[1]["content"]

    def test_system_prompt_states_the_schema(self):
        messages = build_messages(Example("r", ()), include_answer=False)
        system = messages[0]["content"]
        for token in ("aspects", "term", "sentiment", "positive", "conflict"):
            assert token in system


def test_describe_reports_counts():
    examples = {
        "train": [
            Example("a", (("x", "positive"), ("y", "negative"))),
            Example("b", (("z", "neutral"),)),
        ]
    }
    stats = describe(examples)["train"]
    assert stats["reviews"] == 2
    assert stats["aspects"] == 3
    assert stats["max_aspects_in_one_review"] == 2
    assert stats["mean_aspects_per_review"] == pytest.approx(1.5)
    assert stats["sentiment_counts"] == {"negative": 1, "neutral": 1, "positive": 1}
