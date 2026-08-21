"""
CS/AI Expertise Analysis
========================

Relationship between participants' CS/AI background (derived from their field
of work or study) and the key performance metrics (RAIR, RSR, plausibility,
confidence change, final accuracy).

Statistical methods:
1. Descriptive statistics per expertise group (participant-level aggregates)
2. Regression with cluster-robust standard errors (main analysis, trial-level),
   expertise coded as a dummy variable (Non-expert = reference).

Always uses unfiltered data (all participants regardless of job filter).
"""

import os

import pandas as pd

from . import common
from .. import config
from ..config import TUM_BLUE, TUM_ORANGE, TUM_MED_BLUE

JOB_COLUMN = config.DEMOGRAPHIC_COLUMNS["job"]
OUTPUT_FOLDER = config.PLOT_DIRS["cs_expertise"]

CS_AI_KEYWORDS = config.CS_AI_KEYWORDS

EXPERTISE_ORDER = ["Non-expert", "CS/AI Expert"]
EXPERTISE_COLORS = {"CS/AI Expert": TUM_BLUE, "Non-expert": TUM_ORANGE}

BOX_PLOTS = [
    ("RAIR_user", "Reliance on AI when AI is Right (RAIR) by CS/AI Expertise", "expertise_vs_rair.png"),
    ("RSR_user", "Resistance to Wrong AI (RSR) by CS/AI Expertise", "expertise_vs_rsr.png"),
    ("Mean_Plausibility", "Mean Plausibility Rating by CS/AI Expertise", "expertise_vs_plausibility.png"),
    ("Mean_Conf_Delta", "Mean Confidence Change by CS/AI Expertise", "expertise_vs_conf_delta.png"),
    ("Final_Accuracy_User", "Final Accuracy by CS/AI Expertise", "expertise_vs_accuracy.png"),
]


def is_cs_ai_expert(job_text):
    """True if the free-text field of work/study matches a CS/AI keyword."""
    if pd.isna(job_text):
        return False
    job_lower = str(job_text).lower()
    return any(keyword.lower() in job_lower for keyword in CS_AI_KEYWORDS)


def expertise_color(group):
    return EXPERTISE_COLORS.get(group, TUM_MED_BLUE)


def add_expertise_to_long(long_df, df_wide, job_col):
    """Add the participant's field of work and its expertise classification."""
    common.map_to_long(long_df, df_wide, job_col, "job_field")
    long_df["is_expert"] = long_df["job_field"].apply(is_cs_ai_expert).astype(int)
    long_df["expertise_group"] = long_df["is_expert"].map({1: "CS/AI Expert", 0: "Non-expert"})
    return long_df


def _aor_label_offset(index, row):
    """Keep the two group labels from overlapping on the diagonal."""
    if row["expertise_group"] == "CS/AI Expert":
        return ((10, -20), "left", "top")
    return ((10, 15), "left", "bottom")


def main():
    df_wide = common.load_wide_data("CS/AI EXPERTISE ANALYSIS", JOB_COLUMN, ["job", "work", "field"])
    if df_wide is None:
        return
    common.prepare_output_folder(OUTPUT_FOLDER)

    long_df = add_expertise_to_long(common.build_long_data(df_wide), df_wide, JOB_COLUMN)

    print("\nComputing participant-level metrics...")
    participant_metrics = common.compute_participant_level_metrics(
        long_df, df_wide, group_cols=["expertise_group", "is_expert"]
    )
    print(f"Aggregated data for {len(participant_metrics)} participants")

    print("\nExpertise Distribution:")
    print(f"CS/AI Keywords used: {', '.join(CS_AI_KEYWORDS[:5])}... (and {len(CS_AI_KEYWORDS) - 5} more)")
    for group, count in participant_metrics["expertise_group"].value_counts().sort_index().items():
        if pd.notna(group):
            print(f"  {group}: {count} participants")

    common.print_descriptive_statistics(participant_metrics, "expertise_group",
                                        "DESCRIPTIVE STATISTICS BY CS/AI EXPERTISE")

    print("\n" + "=" * 80)
    print("REGRESSION ANALYSES (Main Analysis)")
    print("=" * 80)
    print("Using trial-level data with cluster-robust SEs (same as h_tests.py)")
    common.run_clustered_regressions(
        long_df,
        predictor="is_expert",
        predictor_label="CS/AI_Expertise",
        interpret=common.group_interpreters("CS/AI experts", "non-experts"),
        coding_note="Expertise coded as: Non-expert = 0 (reference), CS/AI Expert = 1",
    )

    print("\nCreating visualizations...")

    for metric, title, filename in BOX_PLOTS:
        common.plot_group_box(participant_metrics, "expertise_group", metric, title,
                              os.path.join(OUTPUT_FOLDER, filename),
                              xlabel="Expertise Group", order=EXPERTISE_ORDER, figsize=(8, 6))

    common.plot_metric_grid(
        participant_metrics, "expertise_group",
        os.path.join(OUTPUT_FOLDER, "expertise_metrics_by_group.png"),
        suptitle="Performance Metrics by CS/AI Expertise", xlabel="Expertise Group",
        order=EXPERTISE_ORDER, numeric_x=False, figsize=(16, 10), xticklabel_fontsize=9,
    )

    print("\nCreating AOR scatter plot...")
    common.plot_aor_by_group(
        long_df, "expertise_group", os.path.join(OUTPUT_FOLDER, "expertise_aor_scatter.png"),
        group_title="CS/AI Expertise", legend_title="Expertise", color_of=expertise_color,
        point_size=400, edge_width=2.5, annotation_fontsize=12, box_pad=0.6, box_linewidth=2.5,
        explanation_fontsize=10, legend_markersize=14, legend_fontsize=11,
        offsets=_aor_label_offset, summary_header="Expertise Group", summary_width=20,
    )

    common.print_completion_summary(
        OUTPUT_FOLDER,
        sections=[
            ("Statistical Analyses Performed", [
                "  1. Descriptive statistics by expertise group",
                "  2. Regression with cluster-robust SEs (main analysis, trial-level)",
                "     - Accounts for repeated measures within participants",
                "     - Same methodology as h_tests.py",
                "     - Expertise as dummy variable (Non-expert = reference)",
            ]),
            ("Expertise Classification", [
                f"  - CS/AI keywords: {', '.join(CS_AI_KEYWORDS[:8])}...",
                "  - Based on job field/study",
            ]),
            ("Visualization Types", [
                "  - Box plots by expertise group",
                "  - Multi-panel violin plots",
                "  - AOR scatter plot by expertise",
            ]),
        ],
        generated_files=[
            ("expertise_vs_rair.png", "box plot"),
            ("expertise_vs_rsr.png", "box plot"),
            ("expertise_vs_plausibility.png", "box plot"),
            ("expertise_vs_conf_delta.png", "box plot"),
            ("expertise_vs_accuracy.png", "box plot"),
            ("expertise_metrics_by_group.png", "multi-panel violin plots"),
            ("expertise_aor_scatter.png", "AOR scatter: RAIR vs RSR"),
        ],
    )


if __name__ == "__main__":
    main()
