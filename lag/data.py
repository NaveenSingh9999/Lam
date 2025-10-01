"""Dataset utilities for the LAG project."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Iterator, List, Tuple
from urllib.request import urlretrieve

import torch
from torch.utils.data import Dataset, DataLoader

from .config import DataConfig, DataSourceConfig
from .tokenizer import BPETokenizer

try:  # Optional heavy dependency
    from datasets import get_dataset_split_names, load_dataset  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - exercised only when datasets is missing
    load_dataset = None
    get_dataset_split_names = None  # type: ignore


TINY_SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def download_tiny_shakespeare(dest_dir: str | Path) -> Path:
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    target_path = dest_dir / "tiny_shakespeare.txt"
    if not target_path.exists():
        urlretrieve(TINY_SHAKESPEARE_URL, target_path)
    return target_path


def read_text_file(path: str | Path) -> str:
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        return fh.read()


def tokenize_corpus(
    tokenizer: BPETokenizer,
    texts: Iterable[str],
    append_eos: bool = False,
) -> torch.Tensor:
    token_ids: list[int] = []
    eos_id = tokenizer.state.token_to_id.get(tokenizer.config.special_tokens[3])
    for text in texts:
        encoded = tokenizer.encode(text, add_special_tokens=False)
        token_ids.extend(encoded)
        if append_eos and eos_id is not None:
            token_ids.append(eos_id)
    if not token_ids:
        raise ValueError("No tokens produced from corpus")
    return torch.tensor(token_ids, dtype=torch.long)


def _format_sample(sample: dict, source: DataSourceConfig) -> str:
    if source.instruction_field and source.response_field:
        instruction = sample.get(source.instruction_field, "").strip()
        response = sample.get(source.response_field, "").strip()
        return f"### Instruction:\n{instruction}\n\n### Response:\n{response}".strip()
    return str(sample.get(source.text_field, "")).strip()


def iterate_hf_source(data_cfg: DataConfig, source: DataSourceConfig) -> Iterator[str]:
    if load_dataset is None:
        raise ImportError(
            "huggingface-datasets is required but not installed. Run `pip install datasets` to use HF sources."
        )
    split = source.split
    try:
        dataset = load_dataset(
            path=source.dataset,
            name=source.subset,
            split=split,
            streaming=data_cfg.streaming,
            cache_dir=data_cfg.cache_dir,
        )
    except ValueError as err:
        if "Unknown split" in str(err) and get_dataset_split_names is not None:
            available = get_dataset_split_names(source.dataset, source.subset)
            if not available:
                raise
            fallback_split = available[0]
            print(
                f"[lag.data] Split '{split}' not found for {source.dataset}. "
                f"Falling back to '{fallback_split}'."
            )
            dataset = load_dataset(
                path=source.dataset,
                name=source.subset,
                split=fallback_split,
                streaming=data_cfg.streaming,
                cache_dir=data_cfg.cache_dir,
            )
        else:
            raise
    if data_cfg.streaming:
        iterator = dataset.shuffle(buffer_size=data_cfg.shuffle_buffer, seed=42)
    else:
        iterator = dataset.shuffle(seed=42) if data_cfg.shuffle_buffer > 0 else dataset
    for idx, row in enumerate(iterator):
        yield _format_sample(row, source)
        if source.max_samples is not None and idx + 1 >= source.max_samples:
            break


def collect_texts_from_sources(data_cfg: DataConfig) -> List[str]:
    texts: list[str] = []
    for source in data_cfg.sources:
        repetitions = max(1, int(round(source.weight)))
        for _ in range(repetitions):
            texts.extend(iterate_hf_source(data_cfg, source))
    return texts


class NextTokenDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    """Creates shifted input/target pairs for autoregressive training."""

    def __init__(self, token_ids: torch.Tensor, seq_len: int) -> None:
        if token_ids.ndim != 1:
            raise ValueError("token_ids must be a 1D tensor")
        if len(token_ids) <= seq_len:
            raise ValueError("token_ids length must exceed seq_len")
        self.token_ids = token_ids
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.token_ids) - self.seq_len

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = idx
        end = idx + self.seq_len
        x = self.token_ids[start:end]
        y = self.token_ids[start + 1 : end + 1]
        return x, y


def build_dataloaders(
    token_ids: torch.Tensor,
    seq_len: int,
    batch_size: int,
    split_ratio: float = 0.9,
    shuffle: bool = True,
) -> tuple[DataLoader, DataLoader]:
    n = len(token_ids)
    split_idx = math.floor(n * split_ratio)
    train_ids = token_ids[:split_idx]
    val_ids = token_ids[split_idx:]
    train_ds = NextTokenDataset(train_ids, seq_len)
    val_ds = NextTokenDataset(val_ids, seq_len)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def build_dataloaders_from_config(
    tokenizer: BPETokenizer,
    data_cfg: DataConfig,
    batch_size: int,
    micro_batch_size: int,
    split_ratio: float = 0.98,
    shuffle: bool = True,
) -> tuple[DataLoader, DataLoader]:
    if not data_cfg.sources:
        raise ValueError("DataConfig.sources is empty; provide at least one dataset source")
    texts = collect_texts_from_sources(data_cfg)
    token_ids = tokenize_corpus(tokenizer, texts, append_eos=True)
    # Use micro-batch for loader; accumulation will reach global batch size
    train_loader, val_loader = build_dataloaders(
        token_ids,
        seq_len=data_cfg.seq_len,
        batch_size=micro_batch_size,
        split_ratio=split_ratio,
        shuffle=shuffle,
    )
    return train_loader, val_loader
