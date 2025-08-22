import torch
from torch.utils.data import Dataset, DataLoader
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# ===============================
# Dataset parameters
# ===============================
repo_id = "lerobot/pusht"   # PushT dataset
history_length = 6           # past states to use
pred_horizon = 64            # future actions to predict
batch_size = 32

# Delta timestamps in seconds
delta_timestamps = {
    "observation.state": [-0.5, -0.4, -0.3, -0.2, -0.1, 0],  # 6 past states
    "action": [t * 0.1 for t in range(pred_horizon)],        # 64 future steps
}

# ===============================
# Load dataset
# ===============================
dataset = LeRobotDataset(
    repo_id,
    delta_timestamps=delta_timestamps,
    video_backend=None,  # disable video decoder
)

# Monkey-patch to completely ignore videos
dataset._query_videos = lambda *args, **kwargs: {}

print(f"Dataset loaded with {dataset.num_frames} frames across {dataset.num_episodes} episodes.")

# ===============================
# Wrapper for BT sequences
# ===============================
class PushTSequenceDataset(Dataset):
    """
    Returns tuples of:
      - past observations [history_length, state_dim]
      - future actions [pred_horizon, action_dim]
    """
    def __init__(self, lerobot_dataset):
        self.dataset = lerobot_dataset
        self.obs_key = "observation.state"
        self.action_key = "action"

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]

        # Get only state + action, ignore image
        obs_seq = data[self.obs_key].float()             # [history_length, state_dim]
        future_actions = data[self.action_key].float()   # [pred_horizon, action_dim]

        return obs_seq, future_actions

# ===============================
# DataLoader
# ===============================
sequence_dataset = PushTSequenceDataset(dataset)
dataloader = DataLoader(
    sequence_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
)

# ===============================
# Test DataLoader
# ===============================
for obs_seq, future_actions in dataloader:
    print(f"obs_seq shape: {obs_seq.shape}")            # [batch, history_length, state_dim]
    print(f"future_actions shape: {future_actions.shape}")  # [batch, pred_horizon, action_dim]
    break
