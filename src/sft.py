"""Supervised instruction fine-tuning with LoRA on a small causal LM.

Two details matter more than anything else here and both are easy to get wrong
silently:

1. **Loss is computed on the response only.** The prompt tokens are masked out
   with -100. Training on the prompt as well teaches the model to reproduce the
   instruction, wastes capacity, and inflates the training loss curve into
   something that looks better than it is.

2. **Padding is masked too.** Pad tokens carry no signal; leaving them in the
   loss makes short examples contribute less than long ones for no reason.

The model is a small instruct checkpoint so the whole thing fits on a 4GB
consumer GPU. LoRA is applied through `peft` here: the from-scratch
implementation lives in the companion classification project, and re-deriving
it would add risk without adding evidence.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

import torch
from torch.utils.data import DataLoader, Dataset

LABEL_IGNORE = -100


@dataclass
class SFTConfig:
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    epochs: int = 3
    batch_size: int = 4
    grad_accum: int = 4
    lr: float = 2e-4
    weight_decay: float = 0.0
    warmup_ratio: float = 0.05
    max_grad_norm: float = 1.0
    max_length: int = 512
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    seed: int = 42
    amp: bool = True


@dataclass
class SFTResult:
    epoch_losses: list[float] = field(default_factory=list)
    trainable_params: int = 0
    total_params: int = 0
    train_seconds: float = 0.0
    peak_vram_mb: float = 0.0
    config: dict | None = None

    def to_dict(self) -> dict:
        return {
            "epoch_losses": self.epoch_losses,
            "trainable_params": self.trainable_params,
            "total_params": self.total_params,
            "trainable_pct": (
                100.0 * self.trainable_params / self.total_params if self.total_params else 0.0
            ),
            "train_seconds": self.train_seconds,
            "peak_vram_mb": self.peak_vram_mb,
            "config": self.config,
        }


class InstructionDataset(Dataset):
    """Tokenised prompt/response pairs with prompt tokens masked from the loss."""

    def __init__(self, examples, tokenizer, max_length: int):
        from .data import build_messages

        self.rows: list[dict] = []
        for example in examples:
            prompt_messages = build_messages(example, include_answer=False)
            prompt_text = tokenizer.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True
            )
            answer_text = example.target_json() + tokenizer.eos_token

            prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
            answer_ids = tokenizer(answer_text, add_special_tokens=False)["input_ids"]

            input_ids = (prompt_ids + answer_ids)[:max_length]
            # Everything before the response is context, not a target.
            labels = ([LABEL_IGNORE] * len(prompt_ids) + answer_ids)[:max_length]

            self.rows.append({"input_ids": input_ids, "labels": labels})

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        return self.rows[i]


def make_collator(pad_token_id: int):
    """Right-pad a batch, masking pad positions out of the loss."""

    def collate(rows: list[dict]) -> dict:
        width = max(len(r["input_ids"]) for r in rows)
        input_ids, labels, attention = [], [], []
        for row in rows:
            pad = width - len(row["input_ids"])
            input_ids.append(row["input_ids"] + [pad_token_id] * pad)
            labels.append(row["labels"] + [LABEL_IGNORE] * pad)
            attention.append([1] * len(row["input_ids"]) + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention, dtype=torch.long),
        }

    return collate


def build_model(cfg: SFTConfig):
    """Load the base model and wrap the attention projections with LoRA."""
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        dtype=torch.float32,  # LoRA params stay fp32; autocast handles the rest
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )
    return get_peft_model(model, lora), tokenizer


def train(model, loader: DataLoader, cfg: SFTConfig, device, verbose: bool = True) -> SFTResult:
    """Fine-tune with gradient accumulation and a linear warmup/decay schedule."""
    torch.manual_seed(cfg.seed)
    model.to(device)
    model.train()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    result = SFTResult(trainable_params=trainable, total_params=total, config=asdict(cfg))

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    steps_per_epoch = max(1, len(loader) // cfg.grad_accum)
    total_steps = steps_per_epoch * cfg.epochs
    warmup = max(1, int(total_steps * cfg.warmup_ratio))

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return step / warmup
        remaining = total_steps - warmup
        return max(0.0, (total_steps - step) / remaining) if remaining > 0 else 0.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    use_amp = cfg.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    start = time.perf_counter()
    for epoch in range(1, cfg.epochs + 1):
        running, counted = 0.0, 0
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(loader, start=1):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = model(**batch).loss
            # Scale down so accumulated gradients average rather than sum.
            scaler.scale(loss / cfg.grad_accum).backward()

            running += loss.item()
            counted += 1

            if step % cfg.grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], cfg.max_grad_norm
                )
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if verbose and step % (cfg.grad_accum * 25) == 0:
                print(
                    f"  epoch {epoch} step {step}/{len(loader)} loss {running / counted:.4f}",
                    flush=True,
                )

        epoch_loss = running / max(1, counted)
        result.epoch_losses.append(epoch_loss)
        if verbose:
            print(f"epoch {epoch}: train loss {epoch_loss:.4f}", flush=True)

    result.train_seconds = time.perf_counter() - start
    if device.type == "cuda":
        result.peak_vram_mb = torch.cuda.max_memory_allocated(device) / 1024**2
    return result
