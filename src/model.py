# src/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class BehaviorTransformer(nn.Module):
    def __init__(self, state_dim=2, action_dim=2, seq_len=32, d_model=128, n_heads=4, n_layers=4, k_bins=16):
        """
        Behavior Transformer (BeT) for PushT.

        Args:
            state_dim: dimension of observation/state
            action_dim: dimension of action
            seq_len: length of input sequence
            d_model: transformer hidden size
            n_heads: number of attention heads
            n_layers: number of transformer layers
            k_bins: number of discretized action bins
        """
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.seq_len = seq_len
        self.k_bins = k_bins

        # Input projection: state -> d_model
        self.input_fc = nn.Linear(state_dim, d_model)

        # Positional embeddings
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, d_model))

        # Transformer encoder (minGPT style)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Bin logits head (categorical over k_bins)
        self.bin_head = nn.Linear(d_model, k_bins)

        # Residual head (predict residual for each bin)
        self.residual_head = nn.Linear(d_model, k_bins * action_dim)

    def forward(self, states):
        """
        states: [batch_size, seq_len, state_dim]
        Returns:
            bin_logits: [batch_size, seq_len, k_bins]
            residuals: [batch_size, seq_len, k_bins, action_dim]
        """
        x = self.input_fc(states)  # [B, seq_len, d_model]
        x = x + self.pos_embedding[:, :self.seq_len, :]  # add positional encoding

        x = self.transformer(x)  # [B, seq_len, d_model]

        bin_logits = self.bin_head(x)  # [B, seq_len, k_bins]
        residuals = self.residual_head(x).view(-1, self.seq_len, self.k_bins, self.action_dim)

        return bin_logits, residuals


# Example usage
if __name__ == "__main__":
    B = 8
    seq_len = 32
    state_dim = 2
    action_dim = 2
    k_bins = 16

    model = BehaviorTransformer(state_dim, action_dim, seq_len, d_model=128, n_heads=4, n_layers=4, k_bins=k_bins)
    dummy_states = torch.randn(B, seq_len, state_dim)
    bin_logits, residuals = model(dummy_states)

    print("Bin logits shape:", bin_logits.shape)  # [B, seq_len, k_bins]
    print("Residuals shape:", residuals.shape)    # [B, seq_len, k_bins, action_dim]
