import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '.')
from h_tests import make_long, compute_rair_rsr_by_age

# Load the data
df_trials = pd.read_excel("experiment_results_with_metrics.xlsx")

# Create long format dataframe
long_df = make_long(df_trials, n_trials=16)

# Compute RAIR and RSR by age groups
results_df = compute_rair_rsr_by_age(df_trials, long_df)

# Save to CSV
if results_df is not None:
    results_df.to_csv("rair_rsr_by_age.csv", index=False)
    print("\n✓ Results saved to: rair_rsr_by_age.csv")

