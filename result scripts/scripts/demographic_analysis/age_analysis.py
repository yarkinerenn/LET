"""
Age Analysis
============

Relationship between participants' age group and the key performance metrics
(RAIR, RSR, plausibility, confidence change, final accuracy).

Statistical methods:
1. Spearman rank correlations (exploratory, participant-level aggregates)
2. Regression with cluster-robust standard errors (main analysis, trial-level):
   OLS for continuous outcomes, logistic regression for binary ones, clustered
   by participant to account for the repeated measures.

Always uses unfiltered data (all participants regardless of job filter).
"""

import os

from . import common
from .. import config
from ..config import TUM_BLUE, TUM_BLUE_DARK, TUM_BLUE_DARKER, TUM_LIGHT_BLUE, TUM_MED_BLUE

AGE_COLUMN = config.DEMOGRAPHIC_COLUMNS["age"]
OUTPUT_FOLDER = config.PLOT_DIRS["age"]

AGE_GROUP_MAPPING = config.AGE_GROUP_MAPPING
AGE_ORDER = ["18-24", "25-34", "35-44", "45-54", "55+"]
AGE_COLORS = [TUM_BLUE_DARKER, TUM_BLUE_DARK, TUM_BLUE, TUM_MED_BLUE, TUM_LIGHT_BLUE]

SCATTER_XLABEL = "Age Group (1=18-24, 2=25-34, 3=35-44, 4=45-54, 5=55+)"

SCATTER_PLOTS = [
    ("RAIR_user", "Reliance on AI when AI is Right (RAIR) by Age Group", "age_vs_rair_scatter.png"),
    ("RSR_user", "Resistance to Wrong AI (RSR) by Age Group", "age_vs_rsr_scatter.png"),
    ("Mean_Plausibility", "Mean Plausibility Rating by Age Group", "age_vs_plausibility_scatter.png"),
    ("Mean_Conf_Delta", "Mean Confidence Change by Age Group", "age_vs_conf_delta_scatter.png"),
    ("Final_Accuracy_User", "Final Accuracy by Age Group", "age_vs_accuracy_scatter.png"),
]

BOX_PLOTS = [
    ("RAIR_user", "RAIR by Age Group", "age_group_vs_rair.png"),
    ("RSR_user", "RSR by Age Group", "age_group_vs_rsr.png"),
    ("Mean_Plausibility", "Mean Plausibility by Age Group", "age_group_vs_plausibility.png"),
    ("Mean_Conf_Delta", "Mean Confidence Change by Age Group", "age_group_vs_conf_delta.png"),
    ("Final_Accuracy_User", "Final Accuracy by Age Group", "age_group_vs_accuracy.png"),
]


def age_color(group):
    """Color of an age group, dark for the youngest and light for the oldest."""
    return AGE_COLORS[AGE_ORDER.index(group)] if group in AGE_ORDER else TUM_MED_BLUE


def add_age_to_long(long_df, df_wide, age_col):
    """Add the participant's age group and its ordinal code to the long data."""
    common.map_to_long(long_df, df_wide, age_col, "age_group")
    long_df["age_ordinal"] = long_df["age_group"].map(AGE_GROUP_MAPPING)
    return long_df


def main():
    df_wide = common.load_wide_data("AGE ANALYSIS", AGE_COLUMN, ["age"])
    if df_wide is None:
        return
    common.prepare_output_folder(OUTPUT_FOLDER)

    long_df = add_age_to_long(common.build_long_data(df_wide), df_wide, AGE_COLUMN)

    print("\nComputing participant-level metrics...")
    participant_metrics = common.compute_participant_level_metrics(
        long_df, df_wide, group_cols=["age_group", "age_ordinal"]
    )
    print(f"Aggregated data for {len(participant_metrics)} participants")

    print("\nAge Group Distribution:")
    for group, count in participant_metrics["age_group"].value_counts().sort_index().items():
        print(f"  {group}: {count} participants")

    # --- Part 1: exploratory correlations ------------------------------------
    print("\n" + "=" * 80)
    print("PART 1: SPEARMAN CORRELATIONS (Exploratory)")
    print("=" * 80)
    print("Performing Spearman rank correlations on participant-level aggregates...")
    print("Using age_ordinal (1-5) for correlation analysis")
    correlations = common.spearman_correlations(participant_metrics, "age_ordinal")
    common.print_correlation_summary(correlations, "AGE CORRELATION ANALYSIS")
    common.save_correlation_summary(correlations, os.path.join(OUTPUT_FOLDER, "age_correlations.csv"))

    # --- Part 2: main analysis -----------------------------------------------
    print("\n" + "=" * 80)
    print("PART 2: REGRESSION ANALYSES (Main Analysis)")
    print("=" * 80)
    print("Using trial-level data with cluster-robust SEs (same as h_tests.py)")
    print("Age groups treated as ordinal (1=18-24, 2=25-34, 3=35-44, 4=45-54, 5=55+)")
    common.run_clustered_regressions(
        long_df,
        predictor="age_ordinal",
        predictor_label="Age",
        interpret=common.ordinal_interpreters("Each 1-level increase in age group"),
        coerce_numeric=True,
    )

    # --- Visualizations ------------------------------------------------------
    print("\nCreating visualizations...")

    for metric, title, filename in SCATTER_PLOTS:
        common.plot_scatter(participant_metrics, "age_ordinal", metric, title,
                            os.path.join(OUTPUT_FOLDER, filename), xlabel=SCATTER_XLABEL)

    for metric, title, filename in BOX_PLOTS:
        common.plot_group_box(participant_metrics, "age_group", metric, title,
                              os.path.join(OUTPUT_FOLDER, filename),
                              xlabel="Age Group", order=AGE_ORDER)

    common.plot_metric_grid(
        participant_metrics, "age_ordinal", os.path.join(OUTPUT_FOLDER, "age_metrics_scatter.png"),
        suptitle="Performance Metrics by Age Group", xlabel="Age Group",
        kind="scatter", trend_range="data", annotate_rho=True,
    )

    print("\nCreating AOR scatter plot...")
    common.plot_aor_by_group(
        long_df, "age_group", os.path.join(OUTPUT_FOLDER, "age_aor_scatter.png"),
        group_title="Age Group", legend_title="Age Group", color_of=age_color,
        order=AGE_ORDER, summary_header="Age Group",
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
            ("Visualization Types", [
                "  - Scatter plots (age as continuous variable)",
                "  - Box plots by age groups",
                "  - Multi-panel scatter plots",
                "  - AOR scatter plot by age groups",
            ]),
        ],
        generated_files=[
            ("age_correlations.csv", "correlation summary"),
            ("age_vs_*_scatter.png", "5 scatter plots"),
            ("age_group_vs_*.png", "5 box plots by age group"),
            ("age_metrics_scatter.png", "multi-panel scatter plots"),
            ("age_aor_scatter.png", "AOR scatter: RAIR vs RSR"),
        ],
    )


if __name__ == "__main__":
    main()
