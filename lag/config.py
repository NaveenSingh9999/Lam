"""Configuration dataclasses for the LAG foundation components."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TokenizerConfig:
    vocab_size: int = 32768
    min_merge_frequency: int = 2
    lowercase: bool = False
    special_tokens: tuple[str, ...] = ("<pad>", "<unk>", "<bos>", "<eos>")


@dataclass
class DataSourceConfig:
    """Metadata describing a single dataset shard used during training."""

    dataset: str
    subset: Optional[str] = None
    split: str = "train"
    text_field: str = "text"
    weight: float = 1.0
    instruction_field: Optional[str] = None
    response_field: Optional[str] = None
    max_samples: Optional[int] = None


@dataclass
class DataConfig:
    sources: list[DataSourceConfig] = field(default_factory=list)
    seq_len: int = 2048
    streaming: bool = False
    pack_sequences: bool = True
    num_workers: int = 2
    shuffle_buffer: int = 10000
    cache_dir: Optional[str] = None


@dataclass
class ModelConfig:
    vocab_size: int
    d_model: int = 512
    n_heads: int = 8
    n_kv_heads: Optional[int] = None
    n_layers: int = 12
    ff_multiplier: float = 4.0
    d_ff: Optional[int] = None
    dropout: float = 0.0
    attention_dropout: float = 0.0
    resid_dropout: float = 0.0
    max_seq_len: int = 2048
    activation: str = "swiglu"
    norm_type: str = "rmsnorm"
    layer_norm_eps: float = 1e-5
    use_bias: bool = False
    use_rotary: bool = True
    rope_theta: float = 10000.0
    rope_scaling: Optional[float] = None
    initializer_range: float = 0.02
    tie_embeddings: bool = True
    gradient_checkpointing: bool = False

    @property
    def kv_heads(self) -> int:
        return self.n_kv_heads or self.n_heads

    @property
    def ff_hidden_size(self) -> int:
        return self.d_ff or int(self.d_model * self.ff_multiplier)


@dataclass
class TrainingConfig:
    batch_size: int = 64
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    max_steps: int = 1000
    eval_interval: int = 200
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0
    scheduler: str = "cosine"
    warmup_steps: int = 0
    warmup_ratio: float = 0.0
    min_lr_ratio: float = 0.1
    device: str = "cpu"


@dataclass
class LAGConfig:
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: Optional[ModelConfig] = None
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def with_vocab_size(self, vocab_size: int) -> "LAGConfig":
        model_cfg = self.model or ModelConfig(vocab_size=vocab_size)
        model_cfg.vocab_size = vocab_size
        return LAGConfig(
            tokenizer=self.tokenizer,
            data=self.data,
            model=model_cfg,
            training=self.training,
        )
