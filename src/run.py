"""Experiment driver: zero-shot baseline, then LoRA instruction tuning.

The comparison the project exists to make is base-instruct-model versus the
same model after instruction tuning on 3k real annotated reviews. Both are
scored by the same function on the same held-out split, so the difference is
attributable to the tuning and not to a changed prompt or metric.

    python -m src.run --limit-eval 300           # quick pass
    python -m src.run --epochs 3                 # full run
    python -m src.run --skip-train                # baseline only
"""

from __future__ import annotations

import argparse
import json
import platform
from datetime import UTC, datetime
from pathlib import Path

ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts"
ADAPTER_DIR = Path(__file__).resolve().parent.parent / "adapters"


def _environment() -> dict:
    import torch

    env = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        env["gpu"] = torch.cuda.get_device_name(0)
        env["vram_total_mb"] = torch.cuda.get_device_properties(0).total_memory / 1024**2
    return env


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--gen-batch-size", type=int, default=8)
    parser.add_argument("--limit-eval", type=int, default=0, help="0 uses the whole test split")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--save-adapter", action="store_true")
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    import torch
    from torch.utils.data import DataLoader

    from .data import describe, load_examples
    from .evaluation import score
    from .generate import generate_batch
    from .sft import InstructionDataset, SFTConfig, build_model, make_collator, train

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)

    examples = load_examples()
    stats = describe(examples)
    print(json.dumps(stats, indent=2), flush=True)

    test_set = examples["test"]
    if args.limit_eval:
        test_set = test_set[: args.limit_eval]
    gold = [e.aspects for e in test_set]

    cfg = SFTConfig(
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lr=args.lr,
        lora_r=args.lora_r,
        max_length=args.max_length,
    )

    record: dict = {
        "model": args.model,
        "dataset": "tomaarsen/setfit-absa-semeval-restaurants (SemEval-2014 Task 4)",
        "corpus": stats,
        "eval_reviews": len(test_set),
        "environment": _environment(),
        "started_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }

    model, tokenizer = build_model(cfg)
    model.to(device)

    # Zero-shot first, while the adapter is still at its zero init and the
    # model is therefore exactly the published checkpoint.
    print("\n=== zero-shot baseline ===", flush=True)
    base_outputs = generate_batch(
        model, tokenizer, test_set, device, batch_size=args.gen_batch_size
    )
    base_scores = score(base_outputs, gold)
    record["zero_shot"] = base_scores.to_dict()
    print(json.dumps(base_scores.to_dict(), indent=2), flush=True)

    if not args.skip_train:
        print("\n=== instruction tuning ===", flush=True)
        dataset = InstructionDataset(examples["train"], tokenizer, cfg.max_length)
        loader = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            collate_fn=make_collator(tokenizer.pad_token_id),
        )
        result = train(model, loader, cfg, device)
        record["training"] = result.to_dict()
        print(json.dumps(result.to_dict(), indent=2), flush=True)

        print("\n=== tuned model ===", flush=True)
        tuned_outputs = generate_batch(
            model, tokenizer, test_set, device, batch_size=args.gen_batch_size
        )
        tuned_scores = score(tuned_outputs, gold)
        record["instruction_tuned"] = tuned_scores.to_dict()
        print(json.dumps(tuned_scores.to_dict(), indent=2), flush=True)

        # Keep a handful of side-by-side examples for the README. Seeing the
        # actual failure mode is more informative than the aggregate.
        record["samples"] = [
            {
                "review": e.text,
                "gold": [{"term": t, "sentiment": s} for t, s in e.aspects],
                "zero_shot": base_outputs[i][:300],
                "tuned": tuned_outputs[i][:300],
            }
            for i, e in enumerate(test_set[:5])
        ]

        if args.save_adapter:
            ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(ADAPTER_DIR)
            tokenizer.save_pretrained(ADAPTER_DIR)
            print(f"adapter saved to {ADAPTER_DIR}", flush=True)

    record["finished_utc"] = datetime.now(UTC).isoformat(timespec="seconds")

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    suffix = f"-{args.tag}" if args.tag else ""
    out = ARTIFACTS / f"absa{suffix}-{stamp}.json"
    out.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
