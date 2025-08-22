# train_bt.py

import torch
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
from dataloader import PushTSequenceDataset, dataset as pushT_dataset  # your dataloader
from model import BehaviorTransformer

# ===============================
# Argument parser
# ===============================
parser = argparse.ArgumentParser()
parser.add_argument("--history_length", type=int, default=6)
parser.add_argument("--pred_horizon", type=int, default=64)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--training_steps", type=int, default=5000)
parser.add_argument("--log_freq", type=int, default=100)
parser.add_argument("--d_model", type=int, default=128)
parser.add_argument("--n_heads", type=int, default=4)
parser.add_argument("--n_layers", type=int, default=4)
parser.add_argument("--k_bins", type=int, default=16)
parser.add_argument("--output_dir", type=str, default="outputs/train/pusht_bet")
args = parser.parse_args()

# ===============================
# Device and output path
# ===============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

# ===============================
# DataLoader
# ===============================
sequence_dataset = PushTSequenceDataset(pushT_dataset)
dataloader = DataLoader(
    sequence_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=0,
    drop_last=True
)

# ===============================
# Model
# ===============================
state_dim = 2
action_dim = 2
seq_len = args.pred_horizon  # Use pred_horizon as seq_len

model = BehaviorTransformer(
    state_dim=state_dim,
    action_dim=action_dim,
    seq_len=seq_len,
    d_model=args.d_model,
    n_heads=args.n_heads,
    n_layers=args.n_layers,
    k_bins=args.k_bins
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
criterion = torch.nn.MSELoss()  # for residual regression

# ===============================
# Training loop
# ===============================
step = 0
done = False

while not done:
    for obs_seq, future_actions in dataloader:
        obs_seq = obs_seq.to(device)                
        future_actions = future_actions.to(device)  

        # Expand obs_seq to match seq_len
        B, H, D = obs_seq.shape
        if H < seq_len:
            repeat_factor = seq_len // H
            remainder = seq_len % H
            obs_seq_exp = obs_seq.repeat(1, repeat_factor, 1)
            if remainder > 0:
                obs_seq_exp = torch.cat([obs_seq_exp, obs_seq[:, :remainder, :]], dim=1)
            obs_seq = obs_seq_exp

        # Forward pass
        bin_logits, residuals = model(obs_seq)

        # Convert bins + residuals to continuous actions
        bin_indices = bin_logits.argmax(dim=-1)
        pred_actions = torch.gather(
            residuals, 2, bin_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, seq_len, 1, action_dim)
        ).squeeze(2)

        pred_actions = pred_actions[:, :args.pred_horizon, :]

        # Compute loss
        loss = criterion(pred_actions, future_actions[:, :args.pred_horizon, :])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % args.log_freq == 0:
            print(f"Step {step}: Loss = {loss.item():.4f}")

        step += 1
        if step >= args.training_steps:
            done = True
            break

# Save trained model
torch.save(model.state_dict(), output_dir / "bet_model.pt")
print(f"Model saved to {output_dir / 'bet_model.pt'}")
