"""
=============================================================
MILESTONE 2: Feature Extraction and Modeling (Weeks 3-4)
=============================================================
"""

import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from prophet import Prophet
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# 1. Feature Extraction
# ─────────────────────────────────────────────
def extract_features(df: pd.DataFrame, window: str = "1h") -> pd.DataFrame:
    print("\n[Feature Extraction] Computing rolling statistical features...")

    numeric_cols = ["heart_rate_bpm", "steps", "spo2_pct", "calories_burned"]
    cols = [c for c in numeric_cols if c in df.columns]

    features = {}
    for col in cols:
        series = df[col]
        roll = series.rolling(window=window, min_periods=1)

        features[f"{col}_mean"]  = roll.mean()
        features[f"{col}_std"]   = roll.std().fillna(0)
        features[f"{col}_min"]   = roll.min()
        features[f"{col}_max"]   = roll.max()
        features[f"{col}_range"] = features[f"{col}_max"] - features[f"{col}_min"]
        features[f"{col}_skew"]  = series.rolling(window=window, min_periods=3).skew().fillna(0)
        features[f"{col}_kurt"]  = series.rolling(window=window, min_periods=4).kurt().fillna(0)
        features[f"{col}_energy"] = roll.apply(lambda x: np.sum(x**2), raw=True)

    feat_df = pd.DataFrame(features, index=df.index)

    # Time features
    feat_df["hour"] = df.index.hour
    feat_df["day_of_week"] = df.index.dayofweek
    feat_df["is_night"] = ((feat_df["hour"] < 6) | (feat_df["hour"] >= 23)).astype(int)

    print(f"    Feature matrix shape: {feat_df.shape}")
    return feat_df


# ─────────────────────────────────────────────
# 2. Prophet Modeling
# ─────────────────────────────────────────────
def prophet_model(df: pd.DataFrame, metric: str, periods: int = 48) -> dict:
    print(f"\n[Prophet] Modeling '{metric}'...")

    prophet_df = df[[metric]].resample("1h").mean().reset_index()
    prophet_df.columns = ["ds", "y"]
    prophet_df["ds"] = prophet_df["ds"].dt.tz_localize(None)
    prophet_df.dropna(inplace=True)

    model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        changepoint_prior_scale=0.05,
        interval_width=0.95,
    )

    model.fit(prophet_df)

    future = model.make_future_dataframe(periods=periods, freq="h")
    forecast = model.predict(future)

    merged = prophet_df.merge(
        forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]],
        on="ds"
    )

    merged["residual"] = merged["y"] - merged["yhat"]
    merged["is_anomaly_prophet"] = (
        (merged["y"] > merged["yhat_upper"]) |
        (merged["y"] < merged["yhat_lower"])
    ).astype(int)

    anomaly_pct = merged["is_anomaly_prophet"].mean() * 100
    print(f"   Prophet anomalies detected: {merged['is_anomaly_prophet'].sum()} ({anomaly_pct:.1f}%)")

    return {"model": model, "forecast": forecast, "residuals": merged}


def plot_prophet(result: dict, metric: str, save_path: str):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    forecast = result["forecast"]
    residuals = result["residuals"]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # Forecast
    axes[0].fill_between(
        forecast["ds"],
        forecast["yhat_lower"],
        forecast["yhat_upper"],
        alpha=0.25
    )
    axes[0].plot(forecast["ds"], forecast["yhat"])
    axes[0].scatter(residuals["ds"], residuals["y"], s=4)

    # Residuals
    axes[1].bar(residuals["ds"], residuals["residual"])

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"    Prophet plot -> {save_path}")


# ─────────────────────────────────────────────
# 3. Clustering (FIXED)
# ─────────────────────────────────────────────
def run_clustering(feat_df: pd.DataFrame, n_clusters: int = 4) -> pd.DataFrame:
    print(f"\n[Clustering] Running KMeans and DBSCAN...")

    # 🔥 FIX: use ALL numeric columns
    cols = feat_df.select_dtypes(include=[np.number]).columns.tolist()
    X = feat_df[cols].fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    feat_df = feat_df.copy()
    feat_df["kmeans_cluster"] = kmeans.fit_predict(X_scaled)

    # DBSCAN
    dbscan = DBSCAN(eps=1.5, min_samples=5)
    feat_df["dbscan_cluster"] = dbscan.fit_predict(X_scaled)
    feat_df["is_outlier_dbscan"] = (feat_df["dbscan_cluster"] == -1).astype(int)

    # 🔥 FIX: Safe PCA
    if X_scaled.shape[1] >= 2:
        pca = PCA(n_components=2)
        p = pca.fit_transform(X_scaled)
        feat_df["pca1"] = p[:, 0]
        feat_df["pca2"] = p[:, 1]
    else:
        feat_df["pca1"] = 0
        feat_df["pca2"] = 0

    print(f"    KMeans clusters: {sorted(feat_df['kmeans_cluster'].unique())}")
    print(f"    DBSCAN outliers: {feat_df['is_outlier_dbscan'].sum()}")

    return feat_df


def plot_clusters(feat_df: pd.DataFrame, save_path: str):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(8, 6))
    plt.scatter(feat_df["pca1"], feat_df["pca2"],
                c=feat_df["kmeans_cluster"], s=5)
    plt.title("Cluster Visualization (PCA)")
    plt.xlabel("PCA1")
    plt.ylabel("PCA2")

    plt.savefig(save_path)
    plt.close()

    print(f"    Cluster plot -> {save_path}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def run_milestone2(df: pd.DataFrame) -> dict:
    print("\n=======================================================")
    print("  MILESTONE 2: Feature Extraction & Modeling")
    print("=======================================================")

    os.makedirs("outputs", exist_ok=True)

    feat_df = extract_features(df)
    feat_df.to_csv("outputs/feature_matrix.csv")
    print("    Feature matrix -> outputs/feature_matrix.csv")

    # Prophet
    prophet_results = {}
    for metric in ["heart_rate_bpm", "steps", "spo2_pct"]:
        if metric in df.columns:
            result = prophet_model(df, metric)
            prophet_results[metric] = result
            plot_prophet(result, metric, f"outputs/prophet_{metric}.png")

    # Clustering
    feat_clustered = run_clustering(feat_df)
    feat_clustered.to_csv("outputs/feature_clustered.csv")
    plot_clusters(feat_clustered, "outputs/milestone2_clusters.png")

    print("\n Milestone 2 Complete!")
    return {"features": feat_clustered, "prophet": prophet_results}


if __name__ == "__main__":
    df = pd.read_csv("outputs/cleaned_data.csv", index_col=0, parse_dates=True)
    run_milestone2(df)