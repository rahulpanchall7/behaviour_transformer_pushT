# src/rollout_visualize.py

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from model import BehaviorTransformer
from dataset_transformer import PushTSequenceDataset
from dataset import load_pusht_parquet
from sklearn.cluster import KMeans

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def rollout_and_visualize(seq_len=32, rollout_steps=100):
    # -------------------
    # Load dataset
    # -------------------
    df = load_pusht_parquet()
    dataset = PushTSequenceDataset(df, seq_len=seq_len)

    # Starting state sequence
    state_seq, _ = dataset[0]
    state_seq = state_seq.unsqueeze(0)  # add batch dimension

    # -------------------
    # KMeans on actions
    # -------------------
    all_actions = np.array(dataset.actions)
    kmeans = KMeans(n_clusters=64, random_state=42).fit(all_actions)

    # -------------------
    # Load trained model
    # -------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BehaviorTransformer(
        state_dim=2,
        action_dim=2,
        seq_len=seq_len,
        d_model=128,
        n_heads=4,
        n_layers=3,
        k_bins=64
    ).to(device)
    model.load_state_dict(torch.load("trained_model.pth", map_location=device))
    model.eval()

    state_seq = state_seq.to(device)

    # -------------------
    # Rollout loop
    # -------------------
    predicted_actions = []

    for t in range(rollout_steps):
        with torch.no_grad():
            bin_logits, residual_preds = model(state_seq)

        # Last timestep prediction
        last_bin_logits = bin_logits[:, -1, :]
        last_residuals = residual_preds[:, -1, :, :]

        chosen_bin = torch.argmax(last_bin_logits, dim=1).item()
        residual = last_residuals[0, chosen_bin, :].cpu().numpy()
        action = kmeans.cluster_centers_[chosen_bin] + residual
        predicted_actions.append(action)

        # Update state_seq (append new state, drop oldest)
        next_state = torch.tensor(action, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        state_seq = torch.cat([state_seq[:, 1:, :], next_state], dim=1)

    predicted_actions = np.array(predicted_actions)
    rollout_file = os.path.join(OUTPUT_DIR, "predicted_rollout.npy")
    np.save(rollout_file, predicted_actions)
    print(f"Saved rollout to {rollout_file}")

    # -------------------
    # Visualization: Predicted vs Ground Truth
    # -------------------
    # Take first sequence from dataset as ground truth
    gt_states, _ = dataset[0]
    gt_states = np.array(gt_states[:rollout_steps])  # truncate to rollout_steps

    plt.figure(figsize=(6,6))
    # Ground truth
    plt.plot(gt_states[:,0], gt_states[:,1], 'g-o', markersize=4, label='Ground Truth')
    plt.scatter(gt_states[0,0], gt_states[0,1], c='green', s=50, label='GT Start')
    plt.scatter(gt_states[-1,0], gt_states[-1,1], c='lime', s=50, label='GT End')

    # Predicted
    plt.plot(predicted_actions[:,0], predicted_actions[:,1], 'b-o', markersize=4, label='Predicted')
    plt.scatter(predicted_actions[0,0], predicted_actions[0,1], c='blue', s=50, label='Pred Start')
    plt.scatter(predicted_actions[-1,0], predicted_actions[-1,1], c='cyan', s=50, label='Pred End')

    plt.title("PushT Rollout: Ground Truth vs Predicted")
    plt.xlabel("Motor 0 / X")
    plt.ylabel("Motor 1 / Y")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")

    plot_file = os.path.join(OUTPUT_DIR, "rollout_comparison_plot.png")
    plt.savefig(plot_file)
    plt.show()
    print(f"Saved comparison plot to {plot_file}")


if __name__ == "__main__":
    rollout_and_visualize(seq_len=32, rollout_steps=100)
