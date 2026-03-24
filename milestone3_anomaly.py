import pandas as pd
import os

def rule_based(df):
    df["rule_anomaly"] = (
        (df["heart_rate_bpm"] > 120) |
        (df["heart_rate_bpm"] < 45) |
        (df["spo2_pct"] < 94)
    ).astype(int)
    return df

def combine(df, feat_df):
    df["dbscan"] = feat_df["is_outlier_dbscan"].reindex(df.index, fill_value=0)
    df["score"] = df["rule_anomaly"] + df["dbscan"]
    df["final"] = (df["score"] > 0).astype(int)
    return df

def run_milestone3(df, feat_df, prophet=None):
    print("\n=== MILESTONE 3 ===")

    os.makedirs("outputs", exist_ok=True)

    df = rule_based(df)
    df = combine(df, feat_df)

    df.to_csv("outputs/anomaly_results.csv")
    print(" Results saved -> outputs/anomaly_results.csv")

    return df

if __name__ == "__main__":
    df = pd.read_csv("outputs/cleaned_data.csv", index_col=0, parse_dates=True)
    feat = pd.read_csv("outputs/feature_clustered.csv", index_col=0, parse_dates=True)
    run_milestone3(df, feat)