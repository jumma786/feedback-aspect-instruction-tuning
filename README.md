# Instruction tuning for aspect-level feedback extraction

![tests](https://github.com/jumma786/feedback-aspect-instruction-tuning/actions/workflows/tests.yml/badge.svg)

Teaching a small instruct model to read a customer review and return structured
aspect-level sentiment as JSON, by LoRA instruction tuning on real human
annotations.

Input:

> *"For some reason, all the seafood on the menu was unavailable except for the Salmon."*

Target:

```json
{"aspects": [
  {"term": "seafood", "sentiment": "negative"},
  {"term": "menu", "sentiment": "negative"},
  {"term": "Salmon", "sentiment": "neutral"}
]}
```

The comparison is the same checkpoint before and after tuning, scored by the
same function on the same held-out reviews, so the difference is attributable to
the tuning rather than to a changed prompt or metric.

## The data, and a problem with it

**SemEval-2014 Task 4** restaurant reviews via
`tomaarsen/setfit-absa-semeval-restaurants`: human-annotated aspect terms, each
with a sentiment label. This is the closest public analogue to what a
customer-experience platform actually extracts from feedback.

Two things had to be handled honestly.

**The source is one row per (review, aspect) pair.** A review mentioning three
things appears three times. Training a generative model on that shape teaches it
to emit one aspect and stop, so rows are regrouped into one example per review
with the full aspect list. Grouping happens *before* splitting, so every aspect
of a review lands in one split; splitting raw rows would put some aspects of a
review in train and others in test.

**The published test split ships unlabelled.** All 1,134 rows carry an empty
sentiment label in this mirror. The loader rejects them rather than passing an
empty string through as a fourth class — which is how the problem surfaced, as a
split that grouped to zero examples instead of a silently corrupted score. The
labelled data is therefore divided three ways with a fixed seed, and the loader
prints a note saying so rather than hiding the substitution.

| Split | Reviews | Annotated aspects | Mean aspects/review |
|---|---:|---:|---:|
| train | 1,415 | 2,591 | 1.83 |
| validation | 201 | 359 | 1.79 |
| test | 403 | 718 | 1.78 |

Verified: **zero text overlap** between the three splits. The `conflict` label
(a reviewer positive and negative about the same thing) is kept rather than
dropped, because collapsing it would quietly inflate sentiment accuracy.

## Scoring: three numbers, not one

Generative extraction fails in three different ways, and one aggregate number
hides which happened:

1. The output is not valid JSON.
2. The JSON is valid but aspects are invented or missed.
3. The aspects are right and the sentiment is wrong.

So `src/evaluation.py` reports them separately:

- **JSON validity**, split into strict (the whole response parsed) and recovered
  (parsed only after stripping a code fence or slicing out the first object). A
  production caller would attempt that recovery, but the strict rate is reported
  too so the recovery cannot hide a real weakness.
- **Aspect precision / recall / F1** on extracted terms, matched
  case-insensitively.
- **Sentiment accuracy over correctly-matched aspects only.**

That last choice is the important one. Computing sentiment over everything would
let a model that extracts almost nothing look excellent. There is a test that
pins it: a model finding 1 of 10 aspects scores 1.00 on sentiment and 0.10 on
recall, so the recall number is what stops the sentiment number being misread.

## Instruction tuning

`src/sft.py`. Two details that are easy to get wrong silently and are both
handled explicitly:

- **Loss is computed on the response only.** Prompt tokens are masked with
  `-100`. Training on the prompt teaches the model to reproduce the instruction,
  wastes capacity, and produces a loss curve that looks better than it is.
- **Padding is masked too**, so short examples are not down-weighted relative to
  long ones for no reason.

Generation is greedy, and the tokeniser is switched to **left padding** for
batched generation — for decoder-only models the continuation starts from the
last position, so right-padding makes shorter prompts generate from pad tokens
and produce nonsense. It is a common silent bug.

LoRA is applied via `peft` here (rank 16 on `q/k/v/o` projections). The
from-scratch LoRA implementation lives in the companion repo,
[`customer-feedback-lora-finetuning`](https://github.com/jumma786/customer-feedback-lora-finetuning);
re-deriving it here would add risk without adding evidence.

## Results

Runs are pending at the time of writing; the zero-shot and tuned scores are
written to `artifacts/` as JSON, alongside side-by-side sample outputs so the
actual failure mode is visible rather than only the aggregate.

```bash
python -m src.run --epochs 3 --save-adapter
```

The expectation worth stating in advance, so it can be checked against the
result: a 0.5B instruct model should be capable of the *format* zero-shot but
weak on *which* spans count as aspects, so the interesting movement should be in
aspect F1 and strict JSON validity rather than in sentiment accuracy.

## Running it

```bash
pip install -r requirements.txt

python -m src.run --limit-eval 100 --skip-train   # quick zero-shot check
python -m src.run --epochs 3 --save-adapter       # full run

pytest -q                                          # 37 tests
```

## Honest limits

- **Restaurant reviews, not a product feedback corpus.** The annotation scheme
  transfers; the vocabulary does not.
- **1,415 training reviews is small** for instruction tuning. It is enough to
  teach format and span selection, not world knowledge.
- **A 4GB GPU caps model size** at around 0.5B parameters. Results would differ
  with a larger base model, and the zero-shot baseline in particular would be
  stronger.
- **Greedy decoding, single seed.** Reproducible, but it does not explore
  whether sampling would do better.
- **`conflict` is rare** (15 instances in test), so per-class sentiment accuracy
  on it is not meaningful.
