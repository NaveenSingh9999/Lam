import torch

from lag.config import ModelConfig
from lag.model import LAGLanguageModel


def test_model_forward_shapes():
    config = ModelConfig(
        vocab_size=128,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=128,
        dropout=0.0,
        max_seq_len=32,
    )
    model = LAGLanguageModel(config)
    inputs = torch.randint(0, config.vocab_size, (2, 16))
    targets = torch.randint(0, config.vocab_size, (2, 16))

    logits, loss = model(inputs, targets)
    assert logits.shape == (2, 16, config.vocab_size)
    assert loss is not None and loss.item() >= 0
