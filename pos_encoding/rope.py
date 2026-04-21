import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt


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
        batch_size, seq_len, num_heads = x.size(0), x.size(1), x.size(2) 
        cos = torch.cos(freqs).unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, head_dim//2]
        sin = torch.sin(freqs).unsqueeze(0).unsqueeze(2)

        # Reshape x to apply rotation
        x1, x2 = x[..., ::2], x[..., 1::2]
        x_stack = torch.stack((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)
        return x_stack.reshape(batch_size, seq_len, num_heads, -1)

    def forward(self, q, k):
        # q, k: [batch_size, seq_len, num_heads, head_dim]
        seq_len = q.size(1)
        freqs = self.freqs[:seq_len]

        # Apply RoPE to q and k
        q_out = self._apply_rotary_emb(q, freqs)
        k_out = self._apply_rotary_emb(k, freqs)

        return q_out, k_out


def visualize_parameter_effects(rope, q, q_out, k, k_out, batch_idx=0, head_idx=0, output_path="pos_encoding/rope_parameter_effects.png"):
    q_in = q[batch_idx, :, head_idx, :].detach().cpu()
    q_rot = q_out[batch_idx, :, head_idx, :].detach().cpu()
    q_delta = (q_rot - q_in).abs()
    k_in = k[batch_idx, :, head_idx, :].detach().cpu()
    k_rot = k_out[batch_idx, :, head_idx, :].detach().cpu()
    freqs = rope.freqs[:q_in.size(0)].detach().cpu()

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    panels = (
        (q_in, "q input embedding", "head_dim"),
        (q_rot, "q after RoPE", "head_dim"),
        (q_delta, "|q_out - q|", "head_dim"),
        (k_in, "k input embedding", "head_dim"),
        (k_rot, "k after RoPE", "head_dim"),
        (freqs, "rotation angle by position/dim pair", "dim pair"),
    )

    for ax, (tensor_map, title, xlabel) in zip(axes.flat, panels):
        image = ax.imshow(tensor_map, aspect="auto", cmap="viridis")
        ax.set_title(f"{title} (batch={batch_idx}, head={head_idx})")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("seq_len")
        fig.colorbar(image, ax=ax)

    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"saved visualization to {output_path}")


def run_example(batch, seq_len, num_heads, head_dim):
    rope = RoPE(head_dim, max_seq_len=seq_len)
    q = torch.rand(batch, seq_len, num_heads, head_dim)
    k = torch.rand(batch, seq_len, num_heads, head_dim)
    q_out, k_out = rope(q, k)
    return rope, q, q_out, k, k_out


def visualize_config_comparison(configs, output_path="pos_encoding/rope_config_comparison.png"):
    fig, axes = plt.subplots(len(configs), 3, figsize=(15, 4 * len(configs)), constrained_layout=True)
    if len(configs) == 1:
        axes = axes.reshape(1, -1)

    for row_axes, config in zip(axes, configs):
        rope, q, q_out, _, _ = run_example(**config)
        q_in = q[0, :, 0, :].detach().cpu()
        q_delta = (q_out[0, :, 0, :] - q[0, :, 0, :]).abs().detach().cpu()
        freqs = rope.freqs[:config["seq_len"]].detach().cpu()
        title_prefix = (
            f"b={config['batch']}, s={config['seq_len']}, "
            f"h={config['num_heads']}, d={config['head_dim']}"
        )

        panels = (
            (q_in, f"{title_prefix}\nq input", "head_dim"),
            (q_delta, f"{title_prefix}\n|q_out - q|", "head_dim"),
            (freqs, f"{title_prefix}\nrotation angles", "dim pair"),
        )

        for ax, (tensor_map, title, xlabel) in zip(row_axes, panels):
            image = ax.imshow(tensor_map, aspect="auto", cmap="viridis")
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("seq_len")
            fig.colorbar(image, ax=ax)

    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"saved visualization to {output_path}")


batch = 30
seq_len = 512
num_heads = 4
head_dim = 100

rope, q, q_out, k, k_out = run_example(batch, seq_len, num_heads, head_dim)
print(q_out.size(), k_out.size())
visualize_parameter_effects(rope, q, q_out, k, k_out)

comparison_configs = [
    {"batch": 1, "seq_len": 128, "num_heads": 2, "head_dim": 32},
    {"batch": 30, "seq_len": 512, "num_heads": 4, "head_dim": 100},
    {"batch": 60, "seq_len": 1024, "num_heads": 8, "head_dim": 128},
    {"batch": 60, "seq_len": 1024, "num_heads": 8, "head_dim": 500},
]
visualize_config_comparison(comparison_configs)
