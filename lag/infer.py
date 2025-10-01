"""Inference script for sampling from a trained LAG model."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import torch

from .config import ModelConfig
from .model import LAGLanguageModel
from .tokenizer import BPETokenizer


def load_model(artifacts_dir: Path, device: torch.device) -> tuple[LAGLanguageModel, BPETokenizer]:
    tokenizer_path = artifacts_dir / "tokenizer.json"
    checkpoint_path = artifacts_dir / "lag_model.pt"

    if not tokenizer_path.exists() or not checkpoint_path.exists():
        raise FileNotFoundError("Expected tokenizer.json and lag_model.pt in artifacts directory")

    tokenizer = BPETokenizer.load(tokenizer_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "config" not in checkpoint or "state_dict" not in checkpoint:
        raise KeyError("Checkpoint missing config/state_dict fields")

    model_cfg = ModelConfig(**checkpoint["config"])
    model = LAGLanguageModel(model_cfg)
    try:
        model.load_state_dict(checkpoint["state_dict"])
    except RuntimeError as err:
        raise RuntimeError(
            "Checkpoint is incompatible with current model architecture. "
            "Please retrain using the updated `lag.train` pipeline to regenerate artifacts."
        ) from err
    model.to(device)
    model.eval()

    return model, tokenizer


def generate(
    prompt: str,
    artifacts_dir: Path,
    max_new_tokens: int,
    device: torch.device,
    temperature: float,
    top_k: Optional[int],
    top_p: Optional[float],
    do_sample: bool,
) -> str:
    model, tokenizer = load_model(artifacts_dir, device)
    encoded = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)

    with torch.no_grad():
        generated_tokens = model.generate(
            encoded,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
        )
    text = tokenizer.decode(generated_tokens[0].tolist())
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text from a trained LAG model")
    parser.add_argument("prompt", type=str, help="Seed text to prime generation")
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"), help="Directory containing tokenizer.json and lag_model.pt")
    parser.add_argument("--max-new-tokens", type=int, default=80, help="Number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature (<=0 for greedy)")
    parser.add_argument("--top-k", type=int, default=0, help="Top-k sampling (0 disables)")
    parser.add_argument("--top-p", type=float, default=0.0, help="Nucleus sampling cumulative probability (0 disables)")
    parser.add_argument("--no-sample", action="store_true", help="Disable stochastic sampling and use greedy decoding")
    parser.add_argument("--device", type=str, default="cpu", help="Device for inference (cpu or cuda)")
    args = parser.parse_args()

    device = torch.device(args.device)
    top_k = args.top_k if args.top_k > 0 else None
    top_p = args.top_p if args.top_p > 0 else None
    completion = generate(
        args.prompt,
        args.artifacts,
        args.max_new_tokens,
        device,
        temperature=args.temperature,
        top_k=top_k,
        top_p=top_p,
        do_sample=not args.no_sample,
    )
    print(completion)


if __name__ == "__main__":  # pragma: no cover
    main()
