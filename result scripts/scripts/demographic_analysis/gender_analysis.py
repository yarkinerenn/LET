"""
Gender Analysis
===============

Relationship between participants' gender and the key performance metrics
(RAIR, RSR, plausibility, confidence change, final accuracy).

Statistical methods:
1. Descriptive statistics per gender (participant-level aggregates)
2. Regression with cluster-robust standard errors (main analysis, trial-level),
   gender coded as a dummy variable (Male = reference, Female = 1).

Always uses unfiltered data (all participants regardless of job filter).
"""

import os

import pandas as pd

from . import common
from .. import config
from ..config import TUM_BLUE, TUM_ORANGE, TUM_GREEN, TUM_MED_BLUE, TUM_GRAY_50

GENDER_COLUMN = config.DEMOGRAPHIC_COLUMNS["gender"]
OUTPUT_FOLDER = config.PLOT_DIRS["gender"]

GENDER_COLORS = {
    "Male": TUM_BLUE,
    "Female": TUM_ORANGE,
    "Non-binary": TUM_GREEN,
    "Prefer not to say": TUM_GRAY_50,
}

BOX_PLOTS = [
    ("RAIR_user", "Reliance on AI when AI is Right (RAIR) by Gender", "gender_vs_rair.png"),
    ("RSR_user", "Resistance to Wrong AI (RSR) by Gender", "gender_vs_rsr.png"),
    ("Mean_Plausibility", "Mean Plausibility Rating by Gender", "gender_vs_plausibility.png"),
    ("Mean_Conf_Delta", "Mean Confidence Change by Gender", "gender_vs_conf_delta.png"),
    ("Final_Accuracy_User", "Final Accuracy by Gender", "gender_vs_accuracy.png"),
]


def gender_color(gender):
    return GENDER_COLORS.get(gender, TUM_MED_BLUE)


def add_gender_to_long(long_df, df_wide, gender_col):
    """Add the participant's gender and the Female dummy to the long data."""
    common.map_to_long(long_df, df_wide, gender_col, "gender")
    long_df["gender_clean"] = long_df["gender"].str.strip().str.lower()
    long_df["is_female"] = (long_df["gender_clean"] == "female").astype(int)
    return long_df


def _aor_label_offset(index, row):
    """Nudge the first annotation below its point so the labels do not overlap."""
    return ((10, -15), "left", "top") if index == 0 else ((10, 10), "left", "bottom")


def main():
    df_wide = common.load_wide_data("GENDER ANALYSIS", GENDER_COLUMN, ["gender"])
    if df_wide is None:
        return
    common.prepare_output_folder(OUTPUT_FOLDER)

    long_df = add_gender_to_long(common.build_long_data(df_wide), df_wide, GENDER_COLUMN)

    print("\nComputing participant-level metrics...")
    participant_metrics = common.compute_participant_level_metrics(
        long_df, df_wide, group_cols=["gender", "is_female"]
    )
    print(f"Aggregated data for {len(participant_metrics)} participants")

    print("\nGender Distribution:")
    for gender, count in participant_metrics["gender"].value_counts().sort_index().items():
        if pd.notna(gender):
            print(f"  {gender}: {count} participants")

    common.print_descriptive_statistics(participant_metrics, "gender", "DESCRIPTIVE STATISTICS BY GENDER")

    print("\n" + "=" * 80)
    print("REGRESSION ANALYSES (Main Analysis)")
    print("=" * 80)
    print("Using trial-level data with cluster-robust SEs (same as h_tests.py)")
    common.run_clustered_regressions(
        long_df,
        predictor="is_female",
        predictor_label="Gender",
        interpret=common.group_interpreters("Females", "males"),
        coding_note="Gender coded as: Male = 0 (reference), Female = 1",
    )

    print("\nCreating visualizations...")

    for metric, title, filename in BOX_PLOTS:
        common.plot_group_box(participant_metrics, "gender", metric, title,
                              os.path.join(OUTPUT_FOLDER, filename),
                              xlabel="Gender", figsize=(8, 6))

    common.plot_metric_grid(
        participant_metrics, "gender", os.path.join(OUTPUT_FOLDER, "gender_metrics_by_group.png"),
        suptitle="Performance Metrics by Gender", xlabel="Gender",
        numeric_x=False, figsize=(16, 10),
    )

    print("\nCreating AOR scatter plot...")
    common.plot_aor_by_group(
        long_df, "gender", os.path.join(OUTPUT_FOLDER, "gender_aor_scatter.png"),
        group_title="Gender", legend_title="Gender", color_of=gender_color,
        point_size=350, annotation_fontsize=11, legend_markersize=12,
        offsets=_aor_label_offset, summary_header="Gender", summary_width=20,
    )

    common.print_completion_summary(
        OUTPUT_FOLDER,
        sections=[
            ("Statistical Analyses Performed", [
                "  1. Descriptive statistics by gender group",
                "  2. Regression with cluster-robust SEs (main analysis, trial-level)",
                "     - Accounts for repeated measures within participants",
                "     - Same methodology as h_tests.py",
                "     - Gender as dummy variable (Male = reference)",
            ]),
            ("Visualization Types", [
                "  - Box plots by gender",
                "  - Multi-panel violin plots",
                "  - AOR scatter plot by gender",
            ]),
        ],
        generated_files=[
            ("gender_vs_rair.png", "box plot"),
            ("gender_vs_rsr.png", "box plot"),
            ("gender_vs_plausibility.png", "box plot"),
            ("gender_vs_conf_delta.png", "box plot"),
            ("gender_vs_accuracy.png", "box plot"),
            ("gender_metrics_by_group.png", "multi-panel violin plots"),
            ("gender_aor_scatter.png", "AOR scatter: RAIR vs RSR"),
        ],
    )


if __name__ == "__main__":
    main()
