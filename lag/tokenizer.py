"""Simple byte-pair encoding tokenizer built from scratch for the LAG project."""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from .config import TokenizerConfig


@dataclass
class BPETokenizerState:
    token_to_id: dict[str, int]
    id_to_token: list[str]
    merges: list[tuple[str, str, str]]
    special_tokens: tuple[str, ...]


class BPETokenizer:
    """Lightweight byte-pair encoding tokenizer with basic persistence."""

    def __init__(self, config: TokenizerConfig | None = None) -> None:
        self.config = config or TokenizerConfig()
        self.state = BPETokenizerState(
            token_to_id={},
            id_to_token=[],
            merges=[],
            special_tokens=self.config.special_tokens,
        )
        for token in self.config.special_tokens:
            self._add_token(token)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def _prepare_text(self, text: str) -> str:
        return text.lower() if self.config.lowercase else text

    def train(self, texts: Iterable[str]) -> None:
        processed_texts = [self._prepare_text(text) for text in texts]
        sequences = [list(text) for text in processed_texts]
        if not sequences:
            raise ValueError("Cannot train tokenizer on empty dataset")

        # Ensure base vocabulary contains all observed characters
        for seq in sequences:
            for char in seq:
                self._add_token(char)

        while len(self.state.token_to_id) < self.config.vocab_size:
            pair_counts = self._compute_pair_statistics(sequences)
            if not pair_counts:
                break
            best_pair, freq = pair_counts.most_common(1)[0]
            if freq < self.config.min_merge_frequency:
                break
            merged_token = "".join(best_pair)
            if merged_token in self.state.token_to_id:
                break
            self.state.merges.append((best_pair[0], best_pair[1], merged_token))
            sequences = [self._merge_sequence(seq, best_pair, merged_token) for seq in sequences]
            self._add_token(merged_token)

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        text = self._prepare_text(text)
        tokens = list(text)
        for left, right, merged in self.state.merges:
            i = 0
            while i < len(tokens) - 1:
                if tokens[i] == left and tokens[i + 1] == right:
                    tokens[i : i + 2] = [merged]
                    if i:
                        i -= 1
                else:
                    i += 1
        if add_special_tokens:
            tokens = [self.config.special_tokens[2]] + tokens + [self.config.special_tokens[3]]
        return [self.state.token_to_id.get(tok, self.state.token_to_id[self.config.special_tokens[1]]) for tok in tokens]

    def decode(self, token_ids: Iterable[int], skip_special_tokens: bool = True) -> str:
        pieces: list[str] = []
        for idx in token_ids:
            if idx < 0 or idx >= len(self.state.id_to_token):
                continue
            token = self.state.id_to_token[idx]
            if skip_special_tokens and token in self.config.special_tokens:
                continue
            pieces.append(token)
        return "".join(pieces)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(
                {
                    "token_to_id": self.state.token_to_id,
                    "id_to_token": self.state.id_to_token,
                    "merges": self.state.merges,
                    "special_tokens": self.state.special_tokens,
                    "config": {
                        "vocab_size": self.config.vocab_size,
                        "min_merge_frequency": self.config.min_merge_frequency,
                        "special_tokens": self.config.special_tokens,
                        "lowercase": self.config.lowercase,
                    },
                },
                fh,
                ensure_ascii=False,
                indent=2,
            )

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        config = TokenizerConfig(
            vocab_size=payload["config"]["vocab_size"],
            min_merge_frequency=payload["config"]["min_merge_frequency"],
            lowercase=payload["config"].get("lowercase", False),
            special_tokens=tuple(payload["config"]["special_tokens"]),
        )
        tokenizer = cls(config=config)
        tokenizer.state = BPETokenizerState(
            token_to_id={k: int(v) for k, v in payload["token_to_id"].items()},
            id_to_token=list(payload["id_to_token"]),
            merges=[tuple(merge) for merge in payload["merges"]],
            special_tokens=tuple(payload["special_tokens"]),
        )
        return tokenizer

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _add_token(self, token: str) -> None:
        if token not in self.state.token_to_id:
            token_id = len(self.state.id_to_token)
            self.state.token_to_id[token] = token_id
            self.state.id_to_token.append(token)

    @staticmethod
    def _merge_sequence(sequence: list[str], pair: tuple[str, str], merged_token: str) -> list[str]:
        i = 0
        result = sequence[:]
        while i < len(result) - 1:
            if result[i] == pair[0] and result[i + 1] == pair[1]:
                result[i : i + 2] = [merged_token]
                if i:
                    i -= 1
            else:
                i += 1
        return result

    @staticmethod
    def _compute_pair_statistics(sequences: list[list[str]]) -> Counter[tuple[str, str]]:
        pair_counts: Counter[tuple[str, str]] = Counter()
        for seq in sequences:
            for i in range(len(seq) - 1):
                pair = (seq[i], seq[i + 1])
                pair_counts[pair] += 1
        return pair_counts
