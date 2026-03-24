import pandas as pd
import numpy as np
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime


# ─────────────────────────────────────────────
# 1. Data Ingestion
# ─────────────────────────────────────────────
def load_data(filepath: str) -> pd.DataFrame:
    ext = os.path.splitext(filepath)[1].lower()
    
    print("Trying to load file from:", os.path.abspath(filepath))  # DEBUG

    if ext == ".csv":
        df = pd.read_csv(filepath)
    elif ext == ".json":
        with open(filepath) as f:
            raw = json.load(f)
        df = pd.DataFrame(raw) if isinstance(raw, list) else pd.DataFrame([raw])
    else:
        raise ValueError(f"Unsupported format: {ext}")

    print(f" Loaded {len(df)} records from {filepath}")

    # 🔥 Column normalization fix
    df.columns = df.columns.str.lower()
    df.rename(columns={
        "heartrate": "heart_rate_bpm",
        "hr": "heart_rate_bpm",
        "spo2": "spo2_pct",
        "oxygen": "spo2_pct"
    }, inplace=True)

    print(" Columns detected:", df.columns.tolist())

    return df


# ─────────────────────────────────────────────
# 2. Timestamp Normalization
# ─────────────────────────────────────────────
def normalize_timestamps(df: pd.DataFrame, ts_col: str = "timestamp") -> pd.DataFrame:
    df = df.copy()
    df[ts_col] = pd.to_datetime(df[ts_col], utc=False)
    df = df.sort_values(ts_col).reset_index(drop=True)
    df.set_index(ts_col, inplace=True)

    print(f" Timestamps normalized. Range: {df.index.min()} -> {df.index.max()}")
    return df


# ─────────────────────────────────────────────
# 3. Missing Value Handling
# ─────────────────────────────────────────────
def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    for col in numeric_cols:
        mask = np.random.random(len(df)) < 0.03
        df.loc[mask, col] = np.nan

    print(f" Missing before: {df.isnull().sum().sum()}")

    df[numeric_cols] = df[numeric_cols].interpolate(method="time", limit_direction="both")
    df.ffill(inplace=True)
    df.bfill(inplace=True)

    print(f" Missing after: {df.isnull().sum().sum()}")
    return df


# ─────────────────────────────────────────────
# 4. Resampling
# ─────────────────────────────────────────────
def resample_data(df: pd.DataFrame, freq: str = "5min") -> pd.DataFrame:
    agg_rules = {
        "heart_rate_bpm": "mean",
        "steps": "sum",
        "sleeping": "max",
        "spo2_pct": "mean",
        "calories_burned": "sum",
        "is_anomaly": "max",
    }

    existing_rules = {k: v for k, v in agg_rules.items() if k in df.columns}
    df_resampled = df[list(existing_rules.keys())].resample(freq).agg(existing_rules)

    df_resampled.ffill(inplace=True)

    print(f" Data resampled. Shape: {df_resampled.shape}")
    return df_resampled


# ─────────────────────────────────────────────
# 5. Visualization
# ─────────────────────────────────────────────
def plot_preprocessed_data(df: pd.DataFrame, save_path: str = "outputs/milestone1_preview.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if "heart_rate_bpm" not in df.columns:
        print(" Skipping plot (missing heart_rate_bpm)")
        return

    sample = df.iloc[:864]

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    axes[0].plot(sample.index, sample.get("heart_rate_bpm", 0))
    axes[1].bar(sample.index, sample.get("steps", 0))
    axes[2].plot(sample.index, sample.get("spo2_pct", 0))

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f" Plot saved -> {save_path}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def run_milestone1(csv_path: str = "fitness_data_raw.csv") -> pd.DataFrame:
    print("\n" + "=" * 55)
    print("  MILESTONE 1: Data Collection & Preprocessing")
    print("=" * 55)

    df_raw = load_data(csv_path)
    df_norm = normalize_timestamps(df_raw)
    df_clean = handle_missing_values(df_norm)
    df_resampled = resample_data(df_clean)

    os.makedirs("outputs", exist_ok=True)
    df_resampled.to_csv("outputs/cleaned_data.csv")

    print(" Cleaned data saved -> outputs/cleaned_data.csv")

    plot_preprocessed_data(df_resampled)

    print("\n Milestone 1 Complete!")
    return df_resampled


if __name__ == "__main__":
    df = run_milestone1("fitness_data_raw.csv")
    print(df.head())