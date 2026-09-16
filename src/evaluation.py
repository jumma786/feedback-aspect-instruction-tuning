"""Scoring for generated aspect extractions.

Generative extraction needs to be scored on more than one axis, because a model
can fail in three quite different ways and a single accuracy number hides which
one happened:

1. It emits something that is not valid JSON at all.
2. It emits valid JSON but invents or misses aspects.
3. It finds the right aspects and gets the sentiment wrong.

So the metrics here are reported separately: JSON validity rate, aspect-level
precision/recall/F1 on the extracted terms, and sentiment accuracy computed
only over the aspects that were correctly identified. Mixing the last two
together would let a model that extracts almost nothing look good on sentiment.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass

# Small models sometimes wrap JSON in prose or a code fence. Recovering that is
# fair, since a production caller would too, but the raw validity rate is
# reported separately so the recovery does not hide a real weakness.
_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class ExtractionScores:
    n: int
    strict_json_rate: float
    recovered_json_rate: float
    aspect_precision: float
    aspect_recall: float
    aspect_f1: float
    sentiment_accuracy_on_matched: float
    exact_match_rate: float
    matched_aspects: int
    predicted_aspects: int
    gold_aspects: int

    def to_dict(self) -> dict:
        return asdict(self)


def parse_prediction(raw: str) -> tuple[list[tuple[str, str]], bool, bool]:
    """Parse a model response into aspects.

    Returns `(aspects, strict_ok, recovered_ok)` where `strict_ok` means the
    whole response parsed as JSON directly, and `recovered_ok` means it parsed
    only after stripping a fence or slicing out the first object.
    """
    text = (raw or "").strip()
    if not text:
        return [], False, False

    def _extract(payload: object) -> list[tuple[str, str]]:
        if not isinstance(payload, dict):
            return []
        items = payload.get("aspects", [])
        if not isinstance(items, list):
            return []
        out: list[tuple[str, str]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            term = str(item.get("term", "")).strip()
            sentiment = str(item.get("sentiment", "")).strip().lower()
            if term:
                out.append((term, sentiment))
        return out

    try:
        return _extract(json.loads(text)), True, True
    except (json.JSONDecodeError, TypeError):
        pass

    for candidate in (_FENCE.search(text), _OBJECT.search(text)):
        if not candidate:
            continue
        body = candidate.group(1) if candidate.re is _FENCE else candidate.group(0)
        try:
            return _extract(json.loads(body)), False, True
        except (json.JSONDecodeError, TypeError):
            continue

    return [], False, False


def _normalise(term: str) -> str:
    """Compare aspect terms case- and whitespace-insensitively.

    Gold terms are spans copied from the review, so "The Staff" and "staff"
    refer to the same thing and should not count as a miss.
    """
    return re.sub(r"\s+", " ", term.strip().lower())


def score(
    predictions: list[str],
    gold: list[tuple[tuple[str, str], ...]],
) -> ExtractionScores:
    """Score raw model outputs against gold aspect lists."""
    if len(predictions) != len(gold):
        raise ValueError(f"length mismatch: {len(predictions)} predictions, {len(gold)} gold")

    strict_ok = recovered_ok = 0
    tp = fp = fn = 0
    sentiment_right = sentiment_total = 0
    exact = 0

    for raw, gold_aspects in zip(predictions, gold, strict=True):
        pred_aspects, is_strict, is_recovered = parse_prediction(raw)
        strict_ok += int(is_strict)
        recovered_ok += int(is_recovered)

        gold_map = {_normalise(t): s for t, s in gold_aspects}
        pred_map = {_normalise(t): s for t, s in pred_aspects}

        matched = set(gold_map) & set(pred_map)
        tp += len(matched)
        fp += len(set(pred_map) - set(gold_map))
        fn += len(set(gold_map) - set(pred_map))

        for term in matched:
            sentiment_total += 1
            sentiment_right += int(pred_map[term] == gold_map[term])

        if pred_map == gold_map and pred_map:
            exact += 1
        elif not pred_map and not gold_map:
            exact += 1

    n = len(predictions)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return ExtractionScores(
        n=n,
        strict_json_rate=strict_ok / n if n else 0.0,
        recovered_json_rate=recovered_ok / n if n else 0.0,
        aspect_precision=precision,
        aspect_recall=recall,
        aspect_f1=f1,
        sentiment_accuracy_on_matched=(
            sentiment_right / sentiment_total if sentiment_total else 0.0
        ),
        exact_match_rate=exact / n if n else 0.0,
        matched_aspects=tp,
        predicted_aspects=tp + fp,
        gold_aspects=tp + fn,
    )
