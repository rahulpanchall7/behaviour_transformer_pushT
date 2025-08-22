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
   git clone <your-repo-url>
   cd <your-repo>
3. **Activate Conda Environment**
   ```bash
   conda activate lerobot
4. **Load the PushT dataset into cache ()**
   ```bash
   python src/dataloader.py
5. **Train the Behavior Transformer model using the command:**
   ```bash
    python src/train.py --history_length 10 --pred_horizon 32 --batch_size 32 --lr 1e-4
6. **Evaluate the trained model and generate output videos:**
    ```bash
   python src/evaluate.py \
    --model_path outputs/train/pusht_bet/bet_model.pt \
    --seq_len 32 \
    --history_length 10 \
    --output_dir outputs/eval/pusht_bet

