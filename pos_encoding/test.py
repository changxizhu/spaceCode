import torch
import torch.nn as nn
from transformer import Transformer


def test_transformer():
    # Model parameters
    vocab_size = 1000
    d_model = 512
    num_heads = 8
    num_layers = 6
    d_ff = 2048
    max_seq_len = 128
    batch_size = 4
    seq_len = 64

    # Create model
    model = Transformer(vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_len)

    # Create random input
    x = torch.randint(0, vocab_size, (batch_size, seq_len))

    print(f"Input shape: {x.shape}")

    # Forward pass
    with torch.no_grad():
        output = model(x)

    print(f"Output shape: {output.shape}")
    print(f"Expected output shape: ({batch_size}, {seq_len}, {vocab_size})")

    # Check if shapes match
    assert output.shape == (batch_size, seq_len, vocab_size), f"Shape mismatch: {output.shape}"

    print("Transformer with RoPE test passed!")

    # Test with different sequence lengths
    seq_len2 = 32
    x2 = torch.randint(0, vocab_size, (batch_size, seq_len2))
    with torch.no_grad():
        output2 = model(x2)
    print(f"Output shape for seq_len={seq_len2}: {output2.shape}")
    assert output2.shape == (batch_size, seq_len2, vocab_size), f"Shape mismatch: {output2.shape}"

    print("All tests passed!")


if __name__ == "__main__":
    test_transformer()