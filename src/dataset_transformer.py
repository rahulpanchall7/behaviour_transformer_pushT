# src/dataset_transformer.py

import torch
from torch.utils.data import Dataset

class PushTDataset(Dataset):
    def __init__(self, df, seq_len=32):
        """
        df: pandas DataFrame for one episode
        seq_len: length of input/output sequences
        """
        self.seq_len = seq_len

        # Convert columns to tensors
        self.states = torch.tensor(df['observation.state'].tolist(), dtype=torch.float32)
        self.actions = torch.tensor(df['action'].tolist(), dtype=torch.float32)

        # Number of sequences
        self.num_seq = len(df) - seq_len

    def __len__(self):
        return self.num_seq

    def __getitem__(self, idx):
        # Sequence of states and actions
        state_seq = self.states[idx:idx+self.seq_len]
        action_seq = self.actions[idx:idx+self.seq_len]
        return state_seq, action_seq

# Example usage
if __name__ == "__main__":
    import pandas as pd
    from dataset import load_pusht_parquet

    df = load_pusht_parquet()
    dataset = PushTDataset(df, seq_len=32)

    print("Dataset length:", len(dataset))
    state_seq, action_seq = dataset[0]
    print("State sequence shape:", state_seq.shape)
    print("Action sequence shape:", action_seq.shape)
