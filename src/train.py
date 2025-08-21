# src/train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.cluster import KMeans
import numpy as np

from dataset_transformer import PushTSequenceDataset
from model import BehaviorTransformer
from dataset import load_pusht_parquet


def train_model():
    # -------------------
    # Load dataset
    # -------------------
    print("Loading dataset...")
    df = load_pusht_parquet()
    dataset = PushTSequenceDataset(df, seq_len=32)
    print(f"Dataset loaded: {len(dataset)} sequences")

    # Train/val split
    val_size = int(0.1 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    print(f"Split dataset: {train_size} train / {val_size} val")

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)

    # -------------------
    # KMeans for action discretization
    # -------------------
    print("Running KMeans on actions...")
    all_actions = np.array(dataset.actions)
    kmeans = KMeans(n_clusters=64, random_state=42).fit(all_actions)
    print("KMeans fitted with 64 clusters")

    # -------------------
    # Define model + optimizer
    # -------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = BehaviorTransformer(
        state_dim=2, 
        action_dim=2,
        seq_len=32,
        d_model=128,
        n_heads=4,
        n_layers=3,
        k_bins=64
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion_ce = nn.CrossEntropyLoss()
    criterion_mse = nn.MSELoss()

    # -------------------
    # Training loop
    # -------------------
    num_epochs = 10
    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        # ---- Train ----
        model.train()
        train_loss = 0.0
        for i, (state_seq, action_seq) in enumerate(train_loader):
            state_seq = state_seq.to(device)
            action_seq = action_seq.to(device)

            # Compute action bins and residuals
            bin_labels = kmeans.predict(action_seq.view(-1, 2).cpu().numpy())
            bin_labels = torch.tensor(bin_labels, dtype=torch.long, device=device)
            residuals = action_seq.view(-1, 2) - torch.tensor(
                kmeans.cluster_centers_[bin_labels.cpu().numpy()],
                dtype=torch.float32,
                device=device
            )

            # Forward
            bin_logits, residual_preds = model(state_seq)

            # Reshape residual_preds: [B*seq_len, k_bins, action_dim]
            residual_preds = residual_preds.view(-1, 64, 2)
            # Select only residuals corresponding to true bin
            chosen_residuals = residual_preds[torch.arange(bin_labels.size(0)), bin_labels]

            # Loss
            loss_ce = criterion_ce(bin_logits.view(-1, 64), bin_labels)
            loss_mse = criterion_mse(chosen_residuals, residuals)
            loss = loss_ce + loss_mse

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if (i+1) % 10 == 0:
                print(f"  Batch {i+1}/{len(train_loader)} - Loss: {loss.item():.4f}")

        avg_train_loss = train_loss / len(train_loader)
        print(f"Average Train Loss: {avg_train_loss:.4f}")

        # ---- Validation ----
        model.eval()
        val_loss = 0.0
        correct_bins = 0
        total_bins = 0

        with torch.no_grad():
            for state_seq, action_seq in val_loader:
                state_seq = state_seq.to(device)
                action_seq = action_seq.to(device)

                bin_labels = kmeans.predict(action_seq.view(-1, 2).cpu().numpy())
                bin_labels = torch.tensor(bin_labels, dtype=torch.long, device=device)
                residuals = action_seq.view(-1, 2) - torch.tensor(
                    kmeans.cluster_centers_[bin_labels.cpu().numpy()],
                    dtype=torch.float32,
                    device=device
                )

                bin_logits, residual_preds = model(state_seq)

                residual_preds = residual_preds.view(-1, 64, 2)
                chosen_residuals = residual_preds[torch.arange(bin_labels.size(0)), bin_labels]

                loss_ce = criterion_ce(bin_logits.view(-1, 64), bin_labels)
                loss_mse = criterion_mse(chosen_residuals, residuals)
                loss = loss_ce + loss_mse
                val_loss += loss.item()

                # Accuracy for bins
                preds = torch.argmax(bin_logits.view(-1, 64), dim=1)
                correct_bins += (preds == bin_labels).sum().item()
                total_bins += bin_labels.size(0)

        avg_val_loss = val_loss / len(val_loader)
        bin_acc = correct_bins / total_bins
        print(f"Validation Loss: {avg_val_loss:.4f}, Bin Accuracy: {bin_acc*100:.2f}%")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "trained_model.pth")
            print("Saved new best model!")

if __name__ == "__main__":
    train_model()
