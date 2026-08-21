"""
NLP Experience Analysis
=======================

Relationship between participants' self-rated NLP experience (1-5 Likert scale)
and the key performance metrics (RAIR, RSR, plausibility, confidence change,
final accuracy).

Statistical methods:
1. Spearman rank correlations (exploratory, participant-level aggregates)
2. Regression with cluster-robust standard errors (main analysis, trial-level).

Always uses unfiltered data (all participants regardless of job filter).
"""

import os

from . import common
from .. import config
from ..config import TUM_BLUE, TUM_BLUE_DARK, TUM_BLUE_DARKER, TUM_LIGHT_BLUE, TUM_MED_BLUE

NLP_EXP_COL = config.DEMOGRAPHIC_COLUMNS["nlp_experience"]
OUTPUT_FOLDER = config.PLOT_DIRS["nlp_experience"]

NLP_LEVELS = [1, 2, 3, 4, 5]
NLP_TICK_LABELS = ["1\n(Novice)", "2", "3\n(Intermediate)", "4", "5\n(Expert)"]
NLP_COLORS = [TUM_BLUE_DARKER, TUM_BLUE_DARK, TUM_BLUE, TUM_MED_BLUE, TUM_LIGHT_BLUE]

BOX_PLOTS = [
    ("RAIR_user", "Reliance on AI when AI is Right (RAIR) by NLP Experience", "nlp_exp_vs_rair.png"),
    ("RSR_user", "Resistance to Wrong AI (RSR) by NLP Experience", "nlp_exp_vs_rsr.png"),
    ("Mean_Plausibility", "Mean Plausibility Rating by NLP Experience", "nlp_exp_vs_plausibility.png"),
    ("Mean_Conf_Delta", "Mean Confidence Change by NLP Experience", "nlp_exp_vs_conf_delta.png"),
    ("Final_Accuracy_User", "Final Accuracy by NLP Experience", "nlp_exp_vs_accuracy.png"),
]


def nlp_color(level):
    return NLP_COLORS[(int(level) - 1) % len(NLP_COLORS)]


def nlp_label(level):
    return f"Level {int(level)}"


def add_nlp_experience_to_long(long_df, df_wide, nlp_col):
    """Add the participant's NLP experience rating to the long data."""
    return common.map_to_long(long_df, df_wide, nlp_col, "nlp_experience")


def _aor_label_offset(index, row):
    """Place each level's label on the side with more free space."""
    level = int(row["nlp_experience"])
    if level <= 2:
        return ((10, -15), "left", "top")
    if level == 3:
        return ((10, 10), "left", "bottom")
    return ((-10, 10), "right", "bottom")


def main():
    df_wide = common.load_wide_data("NLP EXPERIENCE ANALYSIS", NLP_EXP_COL, ["nlp", "experience"])
    if df_wide is None:
        return
    common.prepare_output_folder(OUTPUT_FOLDER)

    long_df = add_nlp_experience_to_long(common.build_long_data(df_wide), df_wide, NLP_EXP_COL)

    print("\nComputing participant-level metrics...")
    participant_metrics = common.compute_participant_level_metrics(
        long_df, df_wide, group_cols=["nlp_experience"]
    )
    print(f"Aggregated data for {len(participant_metrics)} participants")

    print("\nNLP Experience Distribution:")
    for level, count in participant_metrics["nlp_experience"].value_counts().sort_index().items():
        print(f"  Level {int(level)}: {count} participants")

    # --- Part 1: exploratory correlations ------------------------------------
    print("\n" + "=" * 80)
    print("PART 1: SPEARMAN CORRELATIONS (Exploratory)")
    print("=" * 80)
    print("Performing Spearman rank correlations on participant-level aggregates...")
    correlations = common.spearman_correlations(participant_metrics, "nlp_experience")
    common.print_correlation_summary(correlations, "NLP EXPERIENCE CORRELATION ANALYSIS")
    common.save_correlation_summary(
        correlations, os.path.join(OUTPUT_FOLDER, "nlp_experience_correlations.csv")
    )

    # --- Part 2: main analysis -----------------------------------------------
    print("\n" + "=" * 80)
    print("PART 2: REGRESSION ANALYSES (Main Analysis)")
    print("=" * 80)
    print("Using trial-level data with cluster-robust SEs (same as h_tests.py)")
    common.run_clustered_regressions(
        long_df,
        predictor="nlp_experience",
        predictor_label="NLP_Experience",
        interpret=common.ordinal_interpreters("Each 1-point increase in NLP experience"),
        coerce_numeric=True,
    )

    # --- Visualizations ------------------------------------------------------
    print("\nCreating visualizations...")

    for metric, title, filename in BOX_PLOTS:
        common.plot_group_box(participant_metrics, "nlp_experience", metric, title,
                              os.path.join(OUTPUT_FOLDER, filename),
                              xlabel="NLP Experience Level", numeric_x=True,
                              annotate_spearman=True, xlim=(0.5, 5.5),
                              xticks=NLP_LEVELS, xticklabels=NLP_TICK_LABELS)

    common.plot_metric_grid(
        participant_metrics, "nlp_experience",
        os.path.join(OUTPUT_FOLDER, "nlp_exp_metrics_by_level.png"),
        suptitle="Performance Metrics by NLP Experience Level", xlabel="NLP Experience",
        trend_range=(1, 5), annotate_rho=True, xlim=(0.5, 5.5), xticks=NLP_LEVELS,
    )

    print("\nCreating AOR scatter plot...")
    common.plot_aor_by_group(
        long_df, "nlp_experience", os.path.join(OUTPUT_FOLDER, "nlp_exp_aor_scatter.png"),
        group_title="NLP Experience", legend_title="NLP Experience",
        color_of=nlp_color, label_of=nlp_label, point_size=250,
        offsets=_aor_label_offset, summary_title="NLP Experience Level",
        summary_header="Level", summary_label_of=lambda level: int(level), summary_width=8,
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
                "  - Box plots with individual data points (clearer than scatter plots)",
                "  - Violin plots showing full distributions",
                "  - Mean and median lines clearly marked",
                "  - Sample sizes displayed for each experience level",
                "  - Correlation coefficients with significance shown on each plot",
            ]),
        ],
        generated_files=[
            ("nlp_experience_correlations.csv", "correlation summary"),
            ("nlp_exp_vs_rair.png", "box plot"),
            ("nlp_exp_vs_rsr.png", "box plot"),
            ("nlp_exp_vs_plausibility.png", "box plot"),
            ("nlp_exp_vs_conf_delta.png", "box plot"),
            ("nlp_exp_vs_accuracy.png", "box plot"),
            ("nlp_exp_metrics_by_level.png", "multi-panel violin plots"),
            ("nlp_exp_aor_scatter.png", "AOR scatter: RAIR vs RSR"),
        ],
    )


if __name__ == "__main__":
    main()
