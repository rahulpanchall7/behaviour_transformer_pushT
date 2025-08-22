# src/evaluate.py

import argparse
import torch
from pathlib import Path
import gym_pusht  # noqa: F401
import gymnasium as gym
import imageio
import numpy as np

from model import BehaviorTransformer

# ===============================
# Argument parsing
# ===============================
parser = argparse.ArgumentParser(description="Evaluate BehaviorTransformer on PushT environment")
parser.add_argument("--model_path", type=str, default="outputs/train/pusht_bet/bet_model.pt",
                    help="Path to trained model checkpoint")
parser.add_argument("--seq_len", type=int, default=64, help="Sequence length (pred_horizon) for the model")
parser.add_argument("--history_length", type=int, default=6, help="Number of past steps to feed as history")
parser.add_argument("--output_dir", type=str, default="outputs/eval/pusht_bet", help="Directory to save rollout video")
args = parser.parse_args()

# ===============================
# Device and paths
# ===============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
output_directory = Path(args.output_dir)
output_directory.mkdir(parents=True, exist_ok=True)
model_path = Path(args.model_path)

# ===============================
# Hyperparameters
# ===============================
state_dim = 2
action_dim = 2
k_bins = 16
seq_len = args.seq_len
history_length = args.history_length

# ===============================
# Load model
# ===============================
model = BehaviorTransformer(
    state_dim=state_dim,
    action_dim=action_dim,
    seq_len=seq_len,
    d_model=128,
    n_heads=4,
    n_layers=4,
    k_bins=k_bins
).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# ===============================
# Environment
# ===============================
env = gym.make(
    "gym_pusht/PushT-v0",
    obs_type="state",
    max_episode_steps=300
)

# ===============================
# Rollout
# ===============================
history_buffer = []
numpy_observation, info = env.reset(seed=42)
rewards = []
frames = [env.render()]

done = False
step = 0

while not done:
    # Add current state to history
    state_tensor = torch.from_numpy(numpy_observation[:state_dim]).float()
    history_buffer.append(state_tensor)
    if len(history_buffer) > history_length:
        history_buffer.pop(0)

    # Build obs_seq of length seq_len by repeating history
    obs_seq = torch.stack(history_buffer, dim=0)          # [history_length, state_dim]
    repeats = (seq_len + len(obs_seq) - 1) // len(obs_seq)
    obs_seq = obs_seq.repeat(repeats, 1)[:seq_len]        # [seq_len, state_dim]
    obs_seq = obs_seq.unsqueeze(0).to(device)             # [1, seq_len, state_dim]

    # Forward pass
    with torch.inference_mode():
        bin_logits, residuals = model(obs_seq)
        last_bin = bin_logits[:, -1, :].argmax(dim=-1)
        pred_action = residuals[:, -1, last_bin, :]        # [1, action_dim]

    # Step environment
    numpy_action = pred_action.squeeze().cpu().numpy()     # [action_dim]
    numpy_observation, reward, terminated, truncated, info = env.step(numpy_action)

    rewards.append(reward)
    frames.append(env.render())
    done = terminated or truncated
    step += 1

# ===============================
# Save rollout video
# ===============================
fps = env.metadata.get("render_fps", 30)
video_path = output_directory / "rollout_bt.mp4"
imageio.mimsave(str(video_path), np.stack(frames), fps=fps)

print(f"Rollout complete. Total steps: {step}, Total reward: {sum(rewards):.2f}")
print(f"Video saved to: {video_path}")
