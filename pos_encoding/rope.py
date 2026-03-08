import torch
import torch.nn as nn
import math


class RoPE(nn.Module):
    def __init__(self, dim, max_seq_len=2048):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len

        # Precompute the rotation matrices
        self.register_buffer('freqs', self._precompute_freqs(dim, max_seq_len))

    def _precompute_freqs(self, dim, max_seq_len):
        # Compute frequencies for each dimension
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len).float()
        freqs = torch.einsum('i , j -> i j', t, inv_freq)
        return freqs

    def _apply_rotary_emb(self, x, freqs):
        # x: [batch_size, seq_len, num_heads, head_dim]
        # freqs: [seq_len, head_dim//2]
        cos = torch.cos(freqs).unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, head_dim//2]
        sin = torch.sin(freqs).unsqueeze(0).unsqueeze(2)

        # Reshape x to apply rotation
        x1, x2 = x[..., ::2], x[..., 1::2]
        return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)

    def forward(self, q, k):
        # q, k: [batch_size, seq_len, num_heads, head_dim]
        seq_len = q.size(1)
        freqs = self.freqs[:seq_len]

        # Apply RoPE to q and k
        q_out = self._apply_rotary_emb(q, freqs)
        k_out = self._apply_rotary_emb(k, freqs)

        return q_out, k_out