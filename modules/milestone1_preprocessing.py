import pandas as pd
import numpy as np
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────
# 1. Data Ingestion
# ─────────────────────────────────────────────
def load_data(filepath: str) -> pd.DataFrame:
    ext = os.path.splitext(filepath)[1].lower()

    print("Trying to load file from:", os.path.abspath(filepath))

    if ext == ".csv":
        df = pd.read_csv(filepath)
    elif ext == ".json":
        with open(filepath) as f:
            raw = json.load(f)
        df = pd.DataFrame(raw) if isinstance(raw, list) else pd.DataFrame([raw])
    else:
        raise ValueError(f"Unsupported format: {ext}")

    print(f"Loaded {len(df)} records")

    # Normalize column names
    df.columns = df.columns.str.lower()

    # 🔥 IMPORTANT FIX (handles your dataset)
    df.rename(columns={
        "heartrate": "heart_rate_bpm",
        "hr": "heart_rate_bpm",
        "heart_rate": "heart_rate_bpm",   # ✅ ADDED
        "spo2": "spo2_pct",
        "oxygen": "spo2_pct"
    }, inplace=True)

    print("Columns detected:", df.columns.tolist())

    return df


# ─────────────────────────────────────────────
# 2. Timestamp Normalization
# ─────────────────────────────────────────────
def normalize_timestamps(df: pd.DataFrame, ts_col: str = "timestamp") -> pd.DataFrame:
    df = df.copy()
    df[ts_col] = pd.to_datetime(df[ts_col])
    df = df.sort_values(ts_col)
    df.set_index(ts_col, inplace=True)

    print(f"Timestamps range: {df.index.min()} -> {df.index.max()}")
    return df


# ─────────────────────────────────────────────
# 3. Missing Value Handling
# ─────────────────────────────────────────────
def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # simulate missing
    for col in numeric_cols:
        mask = np.random.random(len(df)) < 0.03
        df.loc[mask, col] = np.nan

    print("Missing before:", df.isnull().sum().sum())

    df[numeric_cols] = df[numeric_cols].interpolate(method="time")
    df.ffill(inplace=True)
    df.bfill(inplace=True)

    print("Missing after:", df.isnull().sum().sum())
    return df


# ─────────────────────────────────────────────
# 4. Resampling
# ─────────────────────────────────────────────
def resample_data(df: pd.DataFrame, freq: str = "5min") -> pd.DataFrame:

    # 🔥 Ensure required columns exist
    required_cols = ["heart_rate_bpm", "steps"]

    for col in required_cols:
        if col not in df.columns:
            print(f"Warning: {col} missing")

    agg_rules = {
        "heart_rate_bpm": "mean",
        "steps": "sum",
        "spo2_pct": "mean",
    }

    existing_rules = {k: v for k, v in agg_rules.items() if k in df.columns}

    df_resampled = df[list(existing_rules.keys())].resample(freq).agg(existing_rules)

    df_resampled.interpolate(inplace=True)

    print(f"Data resampled. Shape: {df_resampled.shape}")
    print("Columns after resample:", df_resampled.columns.tolist())

    return df_resampled


# ─────────────────────────────────────────────
# 5. Visualization
# ─────────────────────────────────────────────
def plot_preprocessed_data(df: pd.DataFrame):
    os.makedirs("outputs", exist_ok=True)

    if "heart_rate_bpm" not in df.columns:
        print("Skipping plot (no heart rate)")
        return

    sample = df.iloc[:500]

    plt.figure(figsize=(12,5))
    plt.plot(sample.index, sample["heart_rate_bpm"])
    plt.title("Heart Rate (Preprocessed)")
    plt.savefig("outputs/milestone1_preview.png")
    plt.close()

    print("Plot saved ->outputs/milestone1_preview.png")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def run_milestone1(csv_path: str) -> pd.DataFrame:
    print("\n" + "=" * 50)
    print("MILESTONE 1: Preprocessing")
    print("=" * 50)

    df_raw = load_data(csv_path)
    df_norm = normalize_timestamps(df_raw)
    df_clean = handle_missing_values(df_norm)
    df_resampled = resample_data(df_clean)

    os.makedirs("outputs", exist_ok=True)
    df_resampled.to_csv("outputs/cleaned_data.csv")

    print("Cleaned data saved -> outputs/cleaned_data.csv")

    plot_preprocessed_data(df_resampled)

    print("Milestone 1 Complete!\n")
    return df_resampled


if __name__ == "__main__":
    df = run_milestone1("data/fitness_data_raw.csv")
    print(df.head())