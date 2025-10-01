from lag.config import TokenizerConfig
from lag.tokenizer import BPETokenizer


def test_tokenizer_roundtrip():
    tokenizer = BPETokenizer(TokenizerConfig(vocab_size=64, min_merge_frequency=2))
    tokenizer.train(["hello world", "hello guardian"])

    encoded = tokenizer.encode("hello world")
    decoded = tokenizer.decode(encoded)

    assert "hello" in decoded
    assert "world" in decoded
