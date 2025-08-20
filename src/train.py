# src/train.py

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.cluster import KMeans

from dataset_transformer import PushTDataset
from model import BehaviorTransformer
from dataset import load_pusht_parquet

# ---------------------------
# Hyperparameters
# ---------------------------
SEQ_LEN = 32
BATCH_SIZE = 64
EPOCHS = 20
LR = 3e-4
K_BINS = 16  # number of action clusters

# ---------------------------
# Load dataset
# ---------------------------
df = load_pusht_parquet()
dataset = PushTDataset(df, seq_len=SEQ_LEN)

# Compute k-means clusters for actions
all_actions = np.array(df['action'].tolist())
kmeans = KMeans(n_clusters=K_BINS, random_state=42).fit(all_actions)
bin_centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)

# ---------------------------
# Prepare dataloader
# ---------------------------
def collate_fn(batch):
    state_seqs, action_seqs = zip(*batch)
    states = torch.stack(state_seqs)       # [B, seq_len, state_dim]
    actions = torch.stack(action_seqs)     # [B, seq_len, action_dim]

    # Compute bin indices and residuals
    # [B, seq_len, k_bins] residuals
    bin_indices = []
    residuals = []
    for b in range(actions.shape[0]):
        bins = []
        res = []
        for t in range(actions.shape[1]):
            a = actions[b, t]
            # find closest bin
            dist = torch.norm(bin_centers - a, dim=1)
            bin_idx = torch.argmin(dist)
            bins.append(bin_idx)
            # residuals for all bins
            res_vec = torch.zeros(K_BINS, actions.shape[2])
            res_vec[bin_idx] = a - bin_centers[bin_idx]
            res.append(res_vec)
        bin_indices.append(torch.tensor(bins))
        residuals.append(torch.stack(res))
    bin_indices = torch.stack(bin_indices)     # [B, seq_len]
    residuals = torch.stack(residuals)         # [B, seq_len, K_BINS, action_dim]

    return states, bin_indices, residuals

dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

# ---------------------------
# Model, optimizer, loss
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BehaviorTransformer(state_dim=2, action_dim=2, seq_len=SEQ_LEN, k_bins=K_BINS).to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)

# Focal loss for bin classification
def focal_loss(inputs, targets, gamma=2.0):
    ce_loss = nn.CrossEntropyLoss(reduction='none')
    logpt = -ce_loss(inputs.view(-1, K_BINS), targets.view(-1))
    pt = torch.exp(logpt)
    loss = -((1 - pt) ** gamma) * logpt
    return loss.mean()

# MT-Loss for residuals
def mt_loss(pred_residuals, true_residuals, bin_indices):
    """
    pred_residuals: [B, seq_len, K_BINS, action_dim]
    true_residuals: [B, seq_len, K_BINS, action_dim]
    bin_indices: [B, seq_len]
    """
    B, seq_len, K, action_dim = pred_residuals.shape
    loss = 0.0
    for b in range(B):
        for t in range(seq_len):
            bin_idx = bin_indices[b, t]
            loss += torch.norm(pred_residuals[b, t, bin_idx] - true_residuals[b, t, bin_idx]) ** 2
    return loss / (B * seq_len)

# ---------------------------
# Training loop
# ---------------------------
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0
    for states, bin_indices_batch, residuals_batch in dataloader:
        states = states.to(device)
        bin_indices_batch = bin_indices_batch.to(device)
        residuals_batch = residuals_batch.to(device)

        optimizer.zero_grad()
        bin_logits, pred_residuals = model(states)
        loss_bin = focal_loss(bin_logits, bin_indices_batch)
        loss_res = mt_loss(pred_residuals, residuals_batch, bin_indices_batch)
        loss = loss_bin + loss_res
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch}/{EPOCHS}, Loss: {total_loss / len(dataloader):.6f}")

print("Training finished!")
