"""
Education Level Analysis
========================

Relationship between participants' education level and the key performance
metrics (RAIR, RSR, plausibility, confidence change, final accuracy).

Education is treated as ordinal: high school (1), Bachelor's (2), Master's (3),
PhD or equivalent (4).

Statistical methods:
1. Spearman rank correlations (exploratory, participant-level aggregates)
2. Regression with cluster-robust standard errors (main analysis, trial-level).

Always uses unfiltered data (all participants regardless of job filter).
"""

import os

import numpy as np

from . import common
from .. import config
from ..config import TUM_BLUE_DARK, TUM_BLUE_DARKER, TUM_LIGHT_BLUE, TUM_MED_BLUE

EDU_COLUMN = config.DEMOGRAPHIC_COLUMNS["education"]
OUTPUT_FOLDER = config.PLOT_DIRS["education"]

# Education levels as ordinal numbers
EDU_MAPPING = {
    "High school diploma": 1,
    "Bachelor's degree": 2,
    "Master's degree": 3,
    "PhD or equivalent": 4,
    "Phd or equivalent": 4,  # handle possible case variation
}

EDU_LABELS = {1: "High School", 2: "Bachelor's", 3: "Master's", 4: "PhD"}
EDU_LEVELS = [1, 2, 3, 4]
EDU_COLORS = [TUM_BLUE_DARKER, TUM_BLUE_DARK, TUM_MED_BLUE, TUM_LIGHT_BLUE]

BOX_PLOTS = [
    ("RAIR_user", "Reliance on AI when AI is Right (RAIR) by Education Level", "edu_vs_rair.png"),
    ("RSR_user", "Resistance to Wrong AI (RSR) by Education Level", "edu_vs_rsr.png"),
    ("Mean_Plausibility", "Mean Plausibility Rating by Education Level", "edu_vs_plausibility.png"),
    ("Mean_Conf_Delta", "Mean Confidence Change by Education Level", "edu_vs_conf_delta.png"),
    ("Final_Accuracy_User", "Final Accuracy by Education Level", "edu_vs_accuracy.png"),
]


def edu_color(level):
    return EDU_COLORS[int(level) - 1]


def edu_label(level):
    return EDU_LABELS[int(level)]


def add_education_to_long(long_df, df_wide, edu_col):
    """Add the participant's ordinal education level to the long data."""
    return common.map_to_long(long_df, df_wide, edu_col, "education_level", mapping=EDU_MAPPING)


def _aor_label_offset(index, row):
    """Fan the four level labels out around their points."""
    level = int(row["education_level"])
    if level == 1:
        return ((10, -15), "left", "top")
    if level == 2:
        return ((-15, -15), "right", "top")
    if level == 3:
        return ((10, 10), "left", "bottom")
    return ((-15, 10), "right", "bottom")


def main():
    df_wide = common.load_wide_data("EDUCATION LEVEL ANALYSIS", EDU_COLUMN, ["education"])
    if df_wide is None:
        return
    common.prepare_output_folder(OUTPUT_FOLDER)

    long_df = add_education_to_long(common.build_long_data(df_wide), df_wide, EDU_COLUMN)

    print("\nComputing participant-level metrics...")
    participant_metrics = common.compute_participant_level_metrics(
        long_df, df_wide, group_cols=["education_level"]
    )
    print(f"Aggregated data for {len(participant_metrics)} participants")

    print("\nEducation Level Distribution:")
    for level, count in participant_metrics["education_level"].value_counts().sort_index().items():
        if not np.isnan(level):
            print(f"  {EDU_LABELS.get(int(level), 'Unknown')}: {count} participants")

    # --- Part 1: exploratory correlations ------------------------------------
    print("\n" + "=" * 80)
    print("PART 1: SPEARMAN CORRELATIONS (Exploratory)")
    print("=" * 80)
    print("Performing Spearman rank correlations on participant-level aggregates...")
    correlations = common.spearman_correlations(participant_metrics, "education_level")
    common.print_correlation_summary(correlations, "EDUCATION LEVEL CORRELATION ANALYSIS")
    common.save_correlation_summary(correlations, os.path.join(OUTPUT_FOLDER, "education_correlations.csv"))

    # --- Part 2: main analysis -----------------------------------------------
    print("\n" + "=" * 80)
    print("PART 2: REGRESSION ANALYSES (Main Analysis)")
    print("=" * 80)
    print("Using trial-level data with cluster-robust SEs (same as h_tests.py)")
    common.run_clustered_regressions(
        long_df,
        predictor="education_level",
        predictor_label="Education_Level",
        interpret=common.ordinal_interpreters("Each 1-level increase in education"),
        coerce_numeric=True,
    )

    # --- Visualizations ------------------------------------------------------
    print("\nCreating visualizations...")

    for metric, title, filename in BOX_PLOTS:
        common.plot_group_box(participant_metrics, "education_level", metric, title,
                              os.path.join(OUTPUT_FOLDER, filename),
                              xlabel="Education Level", numeric_x=True, box_width=0.4,
                              jitter_sd=0.06, annotate_spearman=True, xlim=(0.5, 4.5),
                              xticks=EDU_LEVELS, xticklabels=[EDU_LABELS[i] for i in EDU_LEVELS])

    common.plot_metric_grid(
        participant_metrics, "education_level", os.path.join(OUTPUT_FOLDER, "edu_metrics_by_level.png"),
        suptitle="Performance Metrics by Education Level", xlabel="Education Level",
        width=0.5, jitter_sd=0.05, trend_range=(1, 4), annotate_rho=True, xlim=(0.5, 4.5),
        xticks=EDU_LEVELS, xticklabels=[EDU_LABELS[i] for i in EDU_LEVELS], xticklabel_fontsize=9,
    )

    print("\nCreating AOR scatter plot...")
    common.plot_aor_by_group(
        long_df, "education_level", os.path.join(OUTPUT_FOLDER, "edu_aor_scatter.png"),
        group_title="Education Level", legend_title="Education Level",
        color_of=edu_color, label_of=edu_label, offsets=_aor_label_offset,
        summary_header="Level", summary_width=15,
    )

    common.print_completion_summary(
        OUTPUT_FOLDER,
        sections=[
            ("Statistical Analyses Performed", [
                "  1. Spearman correlations (exploratory, participant-level)",
                "  2. Regression with cluster-robust SEs (main analysis, trial-level)",
                "     - Accounts for repeated measures within participants",
                "     - Same methodology as h_tests.py",
            ]),
            ("Visualization Improvements", [
                "  - Box plots with individual data points",
                "  - Violin plots showing full distributions",
                "  - Mean and median lines clearly marked",
                "  - Sample sizes displayed for each education level",
                "  - Correlation coefficients with significance shown on each plot",
            ]),
        ],
        generated_files=[
            ("education_correlations.csv", "correlation summary"),
            ("edu_vs_rair.png", "box plot"),
            ("edu_vs_rsr.png", "box plot"),
            ("edu_vs_plausibility.png", "box plot"),
            ("edu_vs_conf_delta.png", "box plot"),
            ("edu_vs_accuracy.png", "box plot"),
            ("edu_metrics_by_level.png", "multi-panel violin plots"),
            ("edu_aor_scatter.png", "AOR scatter: RAIR vs RSR"),
        ],
    )


if __name__ == "__main__":
    main()
