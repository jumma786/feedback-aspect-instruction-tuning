"""Batch generation for evaluation.

Generation is greedy (`do_sample=False`). For a structured-extraction task the
goal is a reproducible, parseable answer, not a creative one, and sampling would
make the reported metrics depend on a random seed.

Left padding matters here and is a common silent bug: for decoder-only models
the continuation starts from the last position, so right-padding a batch makes
shorter prompts generate from pad tokens and produce nonsense.
"""

from __future__ import annotations

import torch

from .data import Example, build_messages


@torch.no_grad()
def generate_batch(
    model,
    tokenizer,
    examples: list[Example],
    device,
    batch_size: int = 8,
    max_new_tokens: int = 160,
    verbose: bool = True,
) -> list[str]:
    """Return one raw response string per example."""
    model.eval()

    original_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    outputs: list[str] = []
    try:
        for start in range(0, len(examples), batch_size):
            chunk = examples[start : start + batch_size]
            prompts = [
                tokenizer.apply_chat_template(
                    build_messages(example, include_answer=False),
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for example in chunk
            ]
            encoded = tokenizer(
                prompts, return_tensors="pt", padding=True, add_special_tokens=False
            ).to(device)

            generated = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            # Slice off the prompt so only the completion is scored.
            completions = generated[:, encoded["input_ids"].shape[1] :]
            outputs.extend(
                tokenizer.batch_decode(completions, skip_special_tokens=True)
            )

            if verbose and (start // batch_size) % 10 == 0:
                done = min(start + batch_size, len(examples))
                print(f"  generated {done}/{len(examples)}", flush=True)
    finally:
        tokenizer.padding_side = original_side

    return outputs
