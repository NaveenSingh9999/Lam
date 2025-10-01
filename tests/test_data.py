import torch

from lag.data import NextTokenDataset, build_dataloaders


def test_next_token_dataset_shapes():
    token_ids = torch.arange(100, dtype=torch.long)

    dataset = NextTokenDataset(token_ids, seq_len=2)
    x, y = dataset[0]
    assert x.shape == (2,)
    assert y.shape == (2,)

    train_loader, val_loader = build_dataloaders(token_ids, seq_len=2, batch_size=2)
    xb, yb = next(iter(train_loader))
    assert xb.shape[1] == 2
    assert yb.shape[1] == 2
