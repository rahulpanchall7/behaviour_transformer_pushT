# Behavior Transformer for Robotic Manipulation (PushT)

This repository implements the **Behavior Transformer (BeT)** for robotic manipulation on the **PushT** benchmark using the **LeRobot** framework. The goal is to reproduce and evaluate behavior cloning performance using a Transformer-based policy.

---

## 📂 Dataset

We use the **PushT** dataset:  
[https://huggingface.co/datasets/lerobot/pusht](https://huggingface.co/datasets/lerobot/pusht)

---

## 💻 Framework

We use **LeRobot**, a framework for learning robotic behaviors:  
[https://github.com/huggingface/lerobot](https://github.com/huggingface/lerobot)

---

## ⚡ Setup Instructions

1. **Install LeRobot Framework**  
   Follow the installation instructions in the [LeRobot GitHub repo](https://github.com/huggingface/lerobot).
2. **Clone this repository**  
   ```bash
   git clone https://github.com/rahulpanchall7/behaviour_transformer_pushT.git
   cd behaviour_transformer_pushT
3. **Activate Conda Environment and install the dependencies (created in #1)**
   ```bash
   conda activate lerobot
   pip install -r requirements.txt
4. **Login with Huggingface CLI and add the Huggingface API token(you can get it from https://huggingface.co/settings/tokens)**
   ```bash
   huggingface-cli login
6. **Run the dataloader.py script to get the pushT dataset**
   ```bash
   python src/dataloader.py
7. **Train the Behavior Transformer model, for eg:**
   ```bash
    python src/train.py --history_length 10 --pred_horizon 32 --batch_size 32 --lr 1e-4
8. **Evaluate the trained model and generate output video, for eg:**
    ```bash
   python src/evaluate.py \
    --model_path outputs/train/pusht_bet/bet_model.pt \
    --seq_len 32 \
    --history_length 10 \
    --output_dir outputs/eval/pusht_bet



## Report: Behavior Transformer for PushT

### Description of the Implementation
Implemented a **Behavior Transformer (BeT)** for robotic manipulation using the **PushT** benchmark in the **LeRobot** framework.  
- The model takes **past agent states** (`history_length`) as input and predicts a sequence of future actions (`pred_horizon`) using a Transformer-based architecture.  
- **State dimension**: 2 (agent position), **Action dimension**: 2 (velocity commands).  
- The model predicts **discretized action bins** with **residuals** to generate continuous actions.  
- The pipeline supports **training**, **evaluation**, and **video rollout generation**.

### Design Choices and Challenges
**Design Choices:**
- Transformer-based policy to capture temporal dependencies.
- History buffer with sequence repetition for fixed-length Transformer input.
- Bin + residual action prediction for stability.
- Configurable hyperparameters for training and evaluation.

**Challenges:**
- Correctly shaping sequences to match Transformer input (`seq_len` vs `history_length`).
- Ensuring environment-compatible actions.
- Training instability: current configuration did not produce meaningful pushing behavior.

### Results
- **Training** completed without errors, but the policy **did not learn effective pushing behavior**.  
- **Evaluation**: Robot remains mostly stationary and does not push the T object.  
- **Video output**: Rollout video shows limited movement and no task success.  
- **Analysis**: Current hyperparameters, dataset size, and training duration are insufficient for learning meaningful behaviors.
  
<video width="480" controls>
  <source src="outputs/eval/pusht_bet/rollout_bt.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

### Improvements and Scaling
- Increase training steps and dataset coverage.
- Expand model capacity (more layers, larger `d_model`).
- Include richer state information (object positions, velocities).
- Curriculum learning: start with simple tasks, gradually increase difficulty.
- Hyperparameter tuning: `history_length`, `pred_horizon`, `learning rate`, `n_heads`.
- Introduce evaluation metrics such as reward curves or task success rates for better analysis.



