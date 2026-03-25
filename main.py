"""
=============================================================
FitPulse — Health Anomaly Detection (Main Runner)
=============================================================
Runs all 3 milestones using existing dataset.

Usage:
    python main.py              # Run all milestones
    python main.py --m1         # Only Milestone 1
    python main.py --m1 --m2    # Milestones 1 & 2

Launch dashboard:
    streamlit run modules/milestone4_dashboard.py
=============================================================
"""

import argparse
import os
import sys

# Ensure current folder is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import modules
from modules.milestone1_preprocessing import run_milestone1
from modules.milestone2_modeling import run_milestone2
from modules.milestone3_anomaly import run_milestone3


def main():
    parser = argparse.ArgumentParser(description="FitPulse Anomaly Detection Pipeline")
    parser.add_argument("--m1", action="store_true", help="Run Milestone 1")
    parser.add_argument("--m2", action="store_true", help="Run Milestone 2")
    parser.add_argument("--m3", action="store_true", help="Run Milestone 3")
    parser.add_argument("--all", action="store_true", default=True, help="Run all milestones (default)")
    args = parser.parse_args()

    run_all = not any([args.m1, args.m2, args.m3]) or args.all

    print("\n==============================================")
    print("   FitPulse — Health Anomaly Detection System")
    print("==============================================")

    # ── CHECK DATASET ──
    data_path = "data/fitness_data_raw.csv"

    if not os.path.exists(data_path):
        print("\n ERROR: Dataset not found!")
        print("Place your file here:", data_path)
        return
    else:
        print("\nUsing dataset:", data_path)

    # Create outputs folder
    os.makedirs("outputs", exist_ok=True)

    # ── MILESTONE 1 ──
    df_clean = None
    if run_all or args.m1:
        print("\nRunning Milestone 1...")
        df_clean = run_milestone1(data_path)

    # ── MILESTONE 2 ──
    m2_results = None
    if run_all or args.m2:
        print("\nRunning Milestone 2...")
        import pandas as pd
        if df_clean is None:
            df_clean = pd.read_csv("outputs/cleaned_data.csv", index_col=0, parse_dates=True)
        m2_results = run_milestone2(df_clean)

    # ── MILESTONE 3 ──
    if run_all or args.m3:
        print("\nRunning Milestone 3...")
        import pandas as pd
        if df_clean is None:
            df_clean = pd.read_csv("outputs/cleaned_data.csv", index_col=0, parse_dates=True)

        feat_df = m2_results["features"] if m2_results else None
        prophet = m2_results["prophet"] if m2_results else None

        run_milestone3(df_clean, feat_df=feat_df, prophet_results=prophet)

    # ── FINAL MESSAGE ──
    print("\n==============================================")
    print(" ALL MILESTONES COMPLETED")
    print("Outputs saved in: ./outputs/")
    print("==============================================")

    print("\nTo launch dashboard:")
    print(" streamlit run modules/milestone4_dashboard.py\n")


if __name__ == "__main__":
    main()