"""Tests for the extraction scorer.

The scorer decides what every headline number in the README means, so it gets
tested harder than the training code. In particular: a model that outputs
nothing must score zero rather than dividing by zero, and sentiment accuracy
must be computed only over correctly-extracted aspects so that extracting
almost nothing cannot look like a good sentiment score.
"""

from __future__ import annotations

import json

import pytest

from src.evaluation import parse_prediction, score


def _payload(pairs):
    return json.dumps({"aspects": [{"term": t, "sentiment": s} for t, s in pairs]})


class TestParsing:
    def test_plain_json_is_strict(self):
        aspects, strict, recovered = parse_prediction(_payload([("staff", "negative")]))
        assert aspects == [("staff", "negative")]
        assert strict is True
        assert recovered is True

    def test_code_fence_is_recovered_not_strict(self):
        raw = "```json\n" + _payload([("food", "positive")]) + "\n```"
        aspects, strict, recovered = parse_prediction(raw)
        assert aspects == [("food", "positive")]
        assert strict is False
        assert recovered is True

    def test_prose_wrapped_json_is_recovered(self):
        raw = "Here you go: " + _payload([("pizza", "positive")]) + " Hope that helps!"
        aspects, strict, recovered = parse_prediction(raw)
        assert aspects == [("pizza", "positive")]
        assert strict is False
        assert recovered is True

    def test_unparseable_output_scores_nothing(self):
        aspects, strict, recovered = parse_prediction("the staff were rude")
        assert aspects == []
        assert strict is False
        assert recovered is False

    def test_empty_output_is_handled(self):
        assert parse_prediction("") == ([], False, False)
        assert parse_prediction(None) == ([], False, False)

    def test_empty_aspect_list_is_valid(self):
        aspects, strict, _ = parse_prediction('{"aspects": []}')
        assert aspects == []
        assert strict is True

    def test_malformed_items_are_skipped(self):
        raw = (
            '{"aspects": [{"term": "staff", "sentiment": "negative"}, '
            '"junk", {"sentiment": "x"}]}'
        )
        aspects, strict, _ = parse_prediction(raw)
        assert aspects == [("staff", "negative")]
        assert strict is True

    def test_sentiment_is_lowercased(self):
        aspects, _, _ = parse_prediction('{"aspects": [{"term": "food", "sentiment": "POSITIVE"}]}')
        assert aspects == [("food", "positive")]


class TestScoring:
    def test_perfect_prediction(self):
        gold = [(("staff", "negative"), ("food", "positive"))]
        result = score([_payload([("staff", "negative"), ("food", "positive")])], gold)
        assert result.aspect_f1 == pytest.approx(1.0)
        assert result.sentiment_accuracy_on_matched == pytest.approx(1.0)
        assert result.exact_match_rate == pytest.approx(1.0)
        assert result.strict_json_rate == pytest.approx(1.0)

    def test_term_matching_ignores_case_and_spacing(self):
        gold = [(("The Staff", "negative"),)]
        result = score([_payload([("  the staff ", "negative")])], gold)
        assert result.aspect_f1 == pytest.approx(1.0)

    def test_right_aspect_wrong_sentiment_splits_the_metrics(self):
        """Extraction should score 1.0 while sentiment scores 0.0."""
        gold = [(("staff", "negative"),)]
        result = score([_payload([("staff", "positive")])], gold)
        assert result.aspect_f1 == pytest.approx(1.0)
        assert result.sentiment_accuracy_on_matched == pytest.approx(0.0)
        assert result.exact_match_rate == pytest.approx(0.0)

    def test_hallucinated_aspect_costs_precision_only(self):
        gold = [(("staff", "negative"),)]
        result = score([_payload([("staff", "negative"), ("parking", "positive")])], gold)
        assert result.aspect_recall == pytest.approx(1.0)
        assert result.aspect_precision == pytest.approx(0.5)

    def test_missed_aspect_costs_recall_only(self):
        gold = [(("staff", "negative"), ("food", "positive"))]
        result = score([_payload([("staff", "negative")])], gold)
        assert result.aspect_precision == pytest.approx(1.0)
        assert result.aspect_recall == pytest.approx(0.5)

    def test_unparseable_output_scores_zero_without_crashing(self):
        gold = [(("staff", "negative"),)]
        result = score(["not json at all"], gold)
        assert result.aspect_f1 == pytest.approx(0.0)
        assert result.strict_json_rate == pytest.approx(0.0)
        assert result.sentiment_accuracy_on_matched == pytest.approx(0.0)

    def test_empty_gold_and_empty_prediction_is_an_exact_match(self):
        result = score(['{"aspects": []}'], [()])
        assert result.exact_match_rate == pytest.approx(1.0)
        assert result.aspect_f1 == pytest.approx(0.0)  # nothing to find, nothing found

    def test_sentiment_accuracy_ignores_unmatched_aspects(self):
        """A model extracting one right aspect out of ten is not 100% on sentiment."""
        gold = [tuple((f"a{i}", "positive") for i in range(10))]
        result = score([_payload([("a0", "positive")])], gold)
        assert result.sentiment_accuracy_on_matched == pytest.approx(1.0)
        assert result.aspect_recall == pytest.approx(0.1)
        # The recall number is what stops the sentiment score being misread.
        assert result.aspect_f1 < 0.2

    def test_counts_are_reported(self):
        gold = [(("staff", "negative"), ("food", "positive"))]
        result = score([_payload([("staff", "negative"), ("parking", "neutral")])], gold)
        assert result.matched_aspects == 1
        assert result.predicted_aspects == 2
        assert result.gold_aspects == 2

    def test_length_mismatch_is_rejected(self):
        with pytest.raises(ValueError, match="length mismatch"):
            score(["{}"], [(), ()])

    def test_aggregates_over_multiple_examples(self):
        gold = [(("staff", "negative"),), (("food", "positive"),)]
        preds = [_payload([("staff", "negative")]), "garbage"]
        result = score(preds, gold)
        assert result.n == 2
        assert result.strict_json_rate == pytest.approx(0.5)
        assert result.aspect_recall == pytest.approx(0.5)
