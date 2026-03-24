import pandas as pd

# Import all modules
from milestone1_preprocessing import run_milestone1
from milestone2_modeling import run_milestone2
from milestone3_anomaly import run_milestone3


print("\n==============================")
print(" RUNNING COMPLETE PROJECT")
print("==============================")


# ==============================
# MODULE 1
# ==============================
print("\n Running Module 1...")

df = run_milestone1("fitness_data_raw.csv")


# ==============================
# MODULE 2
# ==============================
print("\n Running Module 2...")

results = run_milestone2(df)

feat_df = results["features"]
prophet_results = results["prophet"]


# ==============================
# MODULE 3
# ==============================
print("\n Running Module 3...")

final_df = run_milestone3(df, feat_df, prophet_results)


print("\n==============================")
print(" ALL MODULES COMPLETED!")
print("==============================")