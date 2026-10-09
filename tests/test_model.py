import torch
from gdai.model import ModelConfig, TinyFromScratchTransformer
from gdai.tokenizer import VOCAB_SIZE


def test_transformer_forward_shape():
    cfg = ModelConfig(vocab_size=VOCAB_SIZE, max_seq_len=64, d_model=32, n_layers=2, n_heads=4, dropout=0.0)
    model = TinyFromScratchTransformer(cfg)
    x = torch.randint(0, VOCAB_SIZE, (2, 19))
    logits = model(x)
    assert logits.shape == (2, 19, VOCAB_SIZE)
    assert model.num_parameters() > 0


def test_invalid_attention_dimension_rejected():
    cfg = ModelConfig(max_seq_len=64, d_model=30, n_layers=2, n_heads=4)
    try:
        TinyFromScratchTransformer(cfg)
    except ValueError as exc:
        assert "divisible" in str(exc)
    else:
        raise AssertionError("invalid n_heads/d_model config should fail")
