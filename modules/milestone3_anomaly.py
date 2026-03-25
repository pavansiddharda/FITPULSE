import pandas as pd
import os
import matplotlib.pyplot as plt

# ==========================================================
# RULE-BASED DETECTION
# ==========================================================
def rule_based(df):
    # 🔥 FIX: auto-detect heart rate column
    if "heart_rate_bpm" not in df.columns:
        if "heart_rate" in df.columns:
            df = df.rename(columns={"heart_rate": "heart_rate_bpm"})
        else:
            raise ValueError("No heart rate column found!")

    # 🔥 FIX: ensure SpO2 exists
    if "spo2_pct" not in df.columns:
        df["spo2_pct"] = 97  # default value

    df["rule_anomaly"] = (
        (df["heart_rate_bpm"] > 120) |
        (df["heart_rate_bpm"] < 45) |
        (df["spo2_pct"] < 94)
    ).astype(int)

    return df


# ==========================================================
# COMBINE WITH CLUSTER RESULTS
# ==========================================================
def combine(df, feat_df):
    if feat_df is not None and "is_outlier_dbscan" in feat_df.columns:
        df["dbscan"] = feat_df["is_outlier_dbscan"].reindex(df.index, fill_value=0)
    else:
        df["dbscan"] = 0

    df["score"] = df["rule_anomaly"] + df["dbscan"]
    df["final"] = (df["score"] > 0).astype(int)

    return df


# ==========================================================
# VISUALIZATIONS (IMPORTANT)
# ==========================================================
def plot_results(df):
    os.makedirs("outputs", exist_ok=True)

    sample = df.iloc[:200]

    # --- Heart Rate ---
    plt.figure(figsize=(12,5))
    plt.plot(sample.index, sample["heart_rate_bpm"], label="HR")
    anomaly = sample["final"] == 1
    plt.scatter(sample.index[anomaly], sample["heart_rate_bpm"][anomaly])
    plt.title("Heart Rate Anomaly Detection")
    plt.savefig("outputs/milestone3_heartrate.png")
    plt.close()

    # --- Steps ---
    if "steps" in sample.columns:
        plt.figure(figsize=(12,5))
        plt.plot(sample.index, sample["steps"])
        plt.title("Steps Trend")
        plt.savefig("outputs/milestone3_steps.png")
        plt.close()

    # --- Summary ---
    df["date"] = df.index.date
    daily = df.groupby("date")["final"].mean() * 100

    plt.figure(figsize=(10,4))
    plt.bar(range(len(daily)), daily.values)
    plt.title("Daily Anomaly %")
    plt.savefig("outputs/milestone3_summary.png")
    plt.close()


# ==========================================================
# MAIN FUNCTION
# ==========================================================
def run_milestone3(df, feat_df=None, prophet_results=None):
    print("\n=== MILESTONE 3: Anomaly Detection ===")

    os.makedirs("outputs", exist_ok=True)

    df = rule_based(df)
    df = combine(df, feat_df)

    # Save results
    df.to_csv("outputs/anomaly_results.csv")
    print("Results saved -> outputs/anomaly_results.csv")

    # Generate plots
    plot_results(df)
    print("Plots saved -> outputs/")

    return df


# ==========================================================
# STANDALONE RUN
# ==========================================================
if __name__ == "__main__":
    df = pd.read_csv("outputs/cleaned_data.csv", index_col=0, parse_dates=True)
    feat = pd.read_csv("outputs/feature_matrix.csv", index_col=0, parse_dates=True)
    run_milestone3(df, feat)