"""Build instruction/response pairs from real aspect-annotated customer reviews.

Source: SemEval-2014 Task 4 restaurant reviews, via
`tomaarsen/setfit-absa-semeval-restaurants`. These are human-annotated aspect
terms with a sentiment label each, which is the closest public analogue to what
a customer-experience platform actually extracts from feedback.

The source ships one row per (sentence, aspect) pair. A review mentioning three
things appears three times. Training a generative model on that shape would
teach it to emit one aspect and stop, so the first job here is to regroup rows
back into one example per review with a list of aspects.

Nothing is synthesised. Reviews and labels are used exactly as annotated; the
only transformation is regrouping and rendering to JSON.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from datasets import load_dataset

DATASET = "tomaarsen/setfit-absa-semeval-restaurants"
SEED = 42

# SemEval marks some aspects "conflict" where a reviewer is positive and
# negative about the same thing. It is a real label and is kept rather than
# dropped, because collapsing it would quietly inflate the sentiment scores.
SENTIMENTS = ("positive", "negative", "neutral", "conflict")

SYSTEM_PROMPT = (
    "You extract structured aspect-level sentiment from customer feedback. "
    "Return only JSON matching this schema: "
    '{"aspects": [{"term": string, "sentiment": one of '
    '["positive", "negative", "neutral", "conflict"]}]}. '
    "Use the exact wording from the review for each term. "
    "If the review mentions no specific aspect, return an empty list."
)

USER_TEMPLATE = "Review: {text}"


@dataclass(frozen=True)
class Example:
    """One review and every aspect annotated on it."""

    text: str
    aspects: tuple[tuple[str, str], ...]  # (term, sentiment)

    def target_json(self) -> str:
        payload = {"aspects": [{"term": t, "sentiment": s} for t, s in self.aspects]}
        return json.dumps(payload, ensure_ascii=False)


def _group(split) -> list[Example]:
    """Collapse one-row-per-aspect into one Example per review.

    Order is preserved as annotated so the target JSON is deterministic; a
    model trained against shuffled aspect order would be penalised for
    orderings that are actually correct.
    """
    grouped: dict[str, list[tuple[str, str]]] = {}
    for row in split:
        text = row["text"].strip()
        term = (row["span"] or "").strip()
        label = (row["label"] or "").strip().lower()
        if not text or not term or label not in SENTIMENTS:
            continue
        pairs = grouped.setdefault(text, [])
        if (term, label) not in pairs:
            pairs.append((term, label))

    return [Example(text=t, aspects=tuple(a)) for t, a in grouped.items()]


def load_examples() -> dict[str, list[Example]]:
    """Return grouped train/validation/test example lists.

    Note on splits: this mirror ships its `test` split **unlabelled** - all
    1,134 rows carry an empty sentiment label. It is unusable as an evaluation
    set, so the labelled `train` split is divided three ways instead
    (70/10/20 by review, fixed seed).

    Splitting happens after grouping, so every aspect of a given review lands
    in exactly one split. Splitting the raw rows instead would put some aspects
    of a review in train and others in test, which leaks.
    """
    import random

    raw = load_dataset(DATASET)
    labelled = _group(raw["train"])

    unlabelled = len(raw["test"])
    if unlabelled and not _group(raw["test"]):
        # Surfaced rather than silently swallowed: if the mirror is ever fixed
        # this message stops being printed and the split logic can be revisited.
        print(
            f"note: {DATASET} test split has {unlabelled} rows with no sentiment "
            "label; splitting the labelled train split three ways instead."
        )

    rng = random.Random(SEED)
    rng.shuffle(labelled)

    n = len(labelled)
    n_test = int(n * 0.2)
    n_val = int(n * 0.1)

    return {
        "test": labelled[:n_test],
        "validation": labelled[n_test : n_test + n_val],
        "train": labelled[n_test + n_val :],
    }


def build_messages(example: Example, include_answer: bool) -> list[dict]:
    """Render an example into chat messages.

    `include_answer=False` produces the inference prompt; `True` appends the
    gold response for supervised fine-tuning.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(text=example.text)},
    ]
    if include_answer:
        messages.append({"role": "assistant", "content": example.target_json()})
    return messages


def describe(examples: dict[str, list[Example]]) -> dict:
    """Corpus statistics for the README and run log."""
    stats: dict = {}
    for split, rows in examples.items():
        aspect_counts = [len(e.aspects) for e in rows]
        sentiments: dict[str, int] = {}
        for e in rows:
            for _, s in e.aspects:
                sentiments[s] = sentiments.get(s, 0) + 1
        stats[split] = {
            "reviews": len(rows),
            "aspects": sum(aspect_counts),
            "mean_aspects_per_review": (
                sum(aspect_counts) / len(aspect_counts) if aspect_counts else 0.0
            ),
            "max_aspects_in_one_review": max(aspect_counts) if aspect_counts else 0,
            "sentiment_counts": dict(sorted(sentiments.items())),
        }
    return stats
