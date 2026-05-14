import json
import os
import pandas as pd
import matplotlib.pyplot as plt


SFT_KD_STATE = r"outputs/logit-distilled-tinymistral-2/checkpoint-4375/trainer_state.json"
SOFT_SFT_KD_STATE = r"outputs/logit-distilled-tinymistral-Soft/checkpoint-4375/trainer_state.json"

def load_trainer_logs(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"trainer_state.json not found at: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "log_history" not in data:
        raise ValueError(f"No log_history found in {path}")

    return pd.DataFrame(data["log_history"])

def extract_training_logs(df):
    train_df = df[df["loss"].notna()].copy()

    required_cols = ["step", "loss", "learning_rate"]
    for col in required_cols:
        if col not in train_df.columns:
            raise ValueError(f"Missing column: {col}")

    return train_df

def plot_loss(sft_df, soft_df):
    plt.figure(figsize=(10, 6))

    plt.plot(
        sft_df["step"],
        sft_df["loss"],
        label="SFT + KD Training Loss"
    )

    plt.plot(
        soft_df["step"],
        soft_df["loss"],
        label="Soft-SFT + KD Training Loss"
    )

    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    plt.title("Training Loss Comparison")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_lr(sft_df, soft_df):
    plt.figure(figsize=(10, 6))

    plt.plot(
        sft_df["step"],
        sft_df["learning_rate"],
        label="SFT + KD Learning Rate"
    )

    plt.plot(
        soft_df["step"],
        soft_df["learning_rate"],
        label="Soft-SFT + KD Learning Rate"
    )

    plt.xlabel("Training Steps")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Schedule Comparison")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    sft_logs = load_trainer_logs(SFT_KD_STATE)
    soft_logs = load_trainer_logs(SOFT_SFT_KD_STATE)

    sft_train = extract_training_logs(sft_logs)
    soft_train = extract_training_logs(soft_logs)

    print("SFT + KD Logs Preview:")
    print(sft_train.head())

    print("\nSoft-SFT + KD Logs Preview:")
    print(soft_train.head())

    plot_loss(sft_train, soft_train)
    plot_lr(sft_train, soft_train)