"""Training CLI for the LAG foundation model with advanced configuration."""
from __future__ import annotations

import argparse
import math
from itertools import cycle
from pathlib import Path
from typing import Iterable, List, Optional

import torch
from torch.optim import AdamW
from tqdm import trange

from .config import (
    DataConfig,
    DataSourceConfig,
    LAGConfig,
    ModelConfig,
    TrainingConfig,
    TokenizerConfig,
)
from .data import (
    build_dataloaders,
    collect_texts_from_sources,
    download_tiny_shakespeare,
    read_text_file,
    tokenize_corpus,
)
from .model import LAGLanguageModel
from .tokenizer import BPETokenizer


def evaluate(model: LAGLanguageModel, dataloader: Iterable, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for xb, yb in dataloader:
            xb = xb.to(device)
            yb = yb.to(device)
            _, loss = model(xb, yb)
            assert loss is not None
            total_loss += loss.item() * xb.size(0)
            total_tokens += xb.size(0)
    model.train()
    return total_loss / max(total_tokens, 1)


def compute_learning_rate(step: int, config: TrainingConfig) -> float:
    base_lr = config.learning_rate
    scheduler = config.scheduler.lower()
    warmup_steps = config.warmup_steps or int(config.warmup_ratio * config.max_steps)
    warmup_steps = max(warmup_steps, 1) if scheduler != "none" else 0
    min_lr = base_lr * config.min_lr_ratio

    if scheduler == "none":
        return base_lr

    if step < warmup_steps:
        return base_lr * float(step + 1) / float(max(1, warmup_steps))

    progress = (step - warmup_steps) / float(max(1, config.max_steps - warmup_steps))
    progress = min(max(progress, 0.0), 1.0)

    if scheduler == "cosine":
        return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))

    if scheduler == "linear":
        return base_lr - progress * (base_lr - min_lr)

    raise ValueError(f"Unsupported scheduler: {config.scheduler}")


def parse_source_spec(spec: str) -> DataSourceConfig:
    parts = {}
    for token in spec.split(","):
        if not token.strip():
            continue
        if "=" not in token:
            raise ValueError(f"Invalid hf-source fragment '{token}'. Expected key=value pairs")
        key, value = token.split("=", 1)
        parts[key.strip()] = value.strip()
    if "dataset" not in parts:
        raise ValueError("hf-source specification must include dataset=<path>")
    def maybe_float(name: str) -> Optional[float]:
        if name not in parts:
            return None
        return float(parts[name])

    return DataSourceConfig(
        dataset=parts["dataset"],
        subset=parts.get("subset"),
        split=parts.get("split", "train"),
        text_field=parts.get("text_field", "text"),
        weight=float(parts.get("weight", 1.0)),
        instruction_field=parts.get("instruction_field"),
        response_field=parts.get("response_field"),
        max_samples=int(parts["max_samples"]) if "max_samples" in parts else None,
    )


def train(config: LAGConfig, output_dir: Path, fallback_corpus: Optional[Path] = None) -> None:
    device = torch.device(config.training.device)

    tokenizer = BPETokenizer(config.tokenizer)

    if config.data.sources:
        texts = collect_texts_from_sources(config.data)
    elif fallback_corpus is not None:
        text = read_text_file(fallback_corpus)
        texts = [text]
    else:
        raise ValueError("No data sources provided")

    tokenizer.train(texts)
    vocab_size = len(tokenizer.state.id_to_token)

    if config.model is None:
        model_cfg = ModelConfig(vocab_size=vocab_size)
    else:
        model_cfg = config.model
        model_cfg.vocab_size = vocab_size

    seq_len = config.data.seq_len if config.data.sources else model_cfg.max_seq_len

    token_ids = tokenize_corpus(tokenizer, texts, append_eos=True)
    split_ratio = 0.98 if config.data.sources else 0.9
    train_loader, val_loader = build_dataloaders(
        token_ids,
        seq_len=seq_len,
        batch_size=config.training.micro_batch_size,
        split_ratio=split_ratio,
    )

    model = LAGLanguageModel(model_cfg).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        betas=config.training.betas,
        weight_decay=config.training.weight_decay,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    grad_accum = max(1, config.training.gradient_accumulation_steps)
    global_loader = cycle(train_loader)
    pbar = trange(config.training.max_steps, desc="training", leave=False)

    for step in pbar:
        optimizer.zero_grad()
        step_loss = 0.0
        for _ in range(grad_accum):
            xb, yb = next(global_loader)
            xb = xb.to(device)
            yb = yb.to(device)
            _, loss = model(xb, yb)
            assert loss is not None
            (loss / grad_accum).backward()
            step_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip)
        optimizer.step()

        lr = compute_learning_rate(step, config.training)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        avg_loss = step_loss / grad_accum
        metrics = {"loss": f"{avg_loss:.4f}", "lr": f"{lr:.2e}"}

        if (step + 1) % config.training.eval_interval == 0 or step == config.training.max_steps - 1:
            val_loss = evaluate(model, val_loader, device)
            metrics["val_loss"] = f"{val_loss:.4f}"
        pbar.set_postfix(metrics)

    tokenizer_path = output_dir / "tokenizer.json"
    model_path = output_dir / "lag_model.pt"
    tokenizer.save(tokenizer_path)
    torch.save({"state_dict": model.state_dict(), "config": model_cfg.__dict__}, model_path)


def build_config(args: argparse.Namespace) -> LAGConfig:
    tokenizer_cfg = TokenizerConfig(
        vocab_size=args.vocab_size,
        min_merge_frequency=args.min_merge_frequency,
        lowercase=args.lowercase,
    )

    data_sources: List[DataSourceConfig] = [parse_source_spec(src) for src in args.hf_source]
    data_cfg = DataConfig(
        sources=data_sources,
        seq_len=args.seq_len,
        streaming=args.streaming,
        pack_sequences=not args.no_pack,
        num_workers=args.num_workers,
        shuffle_buffer=args.shuffle_buffer,
        cache_dir=args.hf_cache,
    )

    model_cfg = ModelConfig(
        vocab_size=tokenizer_cfg.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads or None,
        n_layers=args.n_layers,
        ff_multiplier=args.ff_multiplier,
        d_ff=args.d_ff or None,
        dropout=args.dropout,
        attention_dropout=args.attention_dropout,
        resid_dropout=args.resid_dropout,
        max_seq_len=args.seq_len,
        activation=args.activation,
        norm_type=args.norm_type,
        layer_norm_eps=args.layer_norm_eps,
        use_bias=args.use_bias,
        use_rotary=not args.no_rotary,
        rope_theta=args.rope_theta,
        rope_scaling=args.rope_scaling,
        initializer_range=args.initializer_range,
        tie_embeddings=not args.no_tie_embeddings,
        gradient_checkpointing=args.gradient_checkpointing,
    )

    training_cfg = TrainingConfig(
        batch_size=args.batch_size,
        micro_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_steps=args.max_steps,
        eval_interval=args.eval_interval,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2),
        grad_clip=args.grad_clip,
        scheduler=args.scheduler,
        warmup_steps=args.warmup_steps,
        warmup_ratio=args.warmup_ratio,
        min_lr_ratio=args.min_lr_ratio,
        device=args.device,
    )

    return LAGConfig(tokenizer=tokenizer_cfg, data=data_cfg, model=model_cfg, training=training_cfg)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the LAG foundation model from scratch")
    parser.add_argument("--data-dir", type=str, default="data", help="Directory to store/download corpora")
    parser.add_argument("--output-dir", type=str, default="artifacts", help="Directory to store outputs")

    # Tokenizer
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--min-merge-frequency", type=int, default=2)
    parser.add_argument("--lowercase", action="store_true")

    # Data
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--hf-source", action="append", default=[], help=(
        "Add a HuggingFace dataset specification, e.g."
        " --hf-source dataset=HuggingFaceH4/ultrachat_200k,instruction_field=instruction,response_field=response"
    ))
    parser.add_argument("--hf-cache", type=str, default=None, help="Cache directory for HuggingFace datasets")
    parser.add_argument("--streaming", action="store_true", help="Enable dataset streaming mode")
    parser.add_argument("--no-pack", action="store_true", help="Disable sequence packing")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--shuffle-buffer", type=int, default=10000)

    # Model
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--n-kv-heads", type=int, default=0)
    parser.add_argument("--n-layers", type=int, default=12)
    parser.add_argument("--ff-multiplier", type=float, default=4.0)
    parser.add_argument("--d-ff", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--attention-dropout", type=float, default=0.0)
    parser.add_argument("--resid-dropout", type=float, default=0.0)
    parser.add_argument("--activation", type=str, default="swiglu")
    parser.add_argument("--norm-type", type=str, default="rmsnorm")
    parser.add_argument("--layer-norm-eps", type=float, default=1e-5)
    parser.add_argument("--use-bias", action="store_true")
    parser.add_argument("--no-rotary", action="store_true")
    parser.add_argument("--rope-theta", type=float, default=10000.0)
    parser.add_argument("--rope-scaling", type=float, default=0.0)
    parser.add_argument("--initializer-range", type=float, default=0.02)
    parser.add_argument("--no-tie-embeddings", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")

    # Training
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--micro-batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "linear", "none"])
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--warmup-ratio", type=float, default=0.0)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    rope_scaling = args.rope_scaling if args.rope_scaling != 0.0 else None
    args.rope_scaling = rope_scaling

    config = build_config(args)

    fallback_corpus = None
    if not config.data.sources:
        fallback_corpus = download_tiny_shakespeare(data_dir)

    train(config, output_dir, fallback_corpus)


if __name__ == "__main__":  # pragma: no cover
    main()
