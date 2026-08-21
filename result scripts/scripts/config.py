"""
Centralized configuration for the analysis pipeline.

All paths, experiment constants and plot colors live here. Paths are resolved
relative to this file, so the scripts work no matter which directory they are
run from.
"""

from pathlib import Path

# ============================================================================
# PATHS
# ============================================================================

# Root of the "result scripts" project (this file lives in <root>/scripts/)
PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_DIR / "data"
PLOTS_DIR = PROJECT_DIR / "plots"

RAW_DATA_FILES = {
    "prolific_big_small": DATA_DIR / "Prolific-big-small.xlsx",
    "prolific_small_big": DATA_DIR / "Prolific-small-big.xlsx",
    "experiment": DATA_DIR / "experiment.xlsx",
    "experiment2": DATA_DIR / "experiment2.xlsx",
}

PROCESSED_DATA_FILE = DATA_DIR / "experiment_results_with_metrics.xlsx"
PROCESSED_DATA_FILE_CSAI_ONLY = DATA_DIR / "experiment_results_with_metrics_byjob_csai_only.xlsx"
PROCESSED_DATA_FILE_NO_CSAI = DATA_DIR / "experiment_results_with_metrics_byjob_no_csai.xlsx"

OUTPUT_CSV_FILES = {
    "hypothesis_summary": DATA_DIR / "hypothesis_summary_table.csv",
    "participant_aggregates": DATA_DIR / "participant_level_aggregates.csv",
    "within_subjects_faithfulness": DATA_DIR / "within_subjects_comparisons_faithfulness.csv",
    "within_subjects_modelsize": DATA_DIR / "within_subjects_comparisons_modelsize.csv",
}

PLOT_DIRS = {
    "general": PLOTS_DIR / "general",
    "age": PLOTS_DIR / "demographic_analyses" / "age_plots",
    "gender": PLOTS_DIR / "demographic_analyses" / "gender_plots",
    "cs_expertise": PLOTS_DIR / "demographic_analyses" / "cs_expertise_plots",
    "education": PLOTS_DIR / "demographic_analyses" / "education_level_plots",
    "nlp_experience": PLOTS_DIR / "demographic_analyses" / "nlp_experience_plots",
}


def ensure_output_dirs() -> None:
    """Create the data and plot output directories if they do not exist yet."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for plot_dir in PLOT_DIRS.values():
        plot_dir.mkdir(parents=True, exist_ok=True)


# ============================================================================
# EXPERIMENT CONFIGURATION
# ============================================================================

N_TRIALS = 16
BLOCK_SIZE = 5
START_COL_INDEX = 8  # 0-based index of the first trial column
SHEET_INDEX = 0

# Per-question ground truth, AI prediction and faithfulness labels.
# Each string has one character per trial (length == N_TRIALS).
GT_STRING = "DTDTDTDTTDTDTDTD"     # Ground truth (D/T)
AI_STRING = "DTTDDTTDTDDTTDDT"     # AI prediction (D/T)
FAITH_STRING = "FUFUFUFUFUFUFUFU"  # Faithfulness (F/U)

GT_LABELS = list(GT_STRING)
AI_LABELS = list(AI_STRING)
FAITH_LABELS = list(FAITH_STRING)

# Which raw file belongs to which form (A: big model first, B: small model first)
FILE_FORM_MAPPING = [
    (RAW_DATA_FILES["prolific_big_small"], "A"),
    (RAW_DATA_FILES["prolific_small_big"], "B"),
    (RAW_DATA_FILES["experiment"], "A"),
    (RAW_DATA_FILES["experiment2"], "B"),
]

# ============================================================================
# PARTICIPANT FILTERING (data processing only)
# ============================================================================

FILTER_BY_JOB = False
JOB_COLUMN = "What is your field of work/study ?"
FILTER_MODE = "include"  # "include" = only CS/AI, "exclude" = everyone but CS/AI

CS_AI_KEYWORDS = [
    "computer science",
    "computer",
    "ai",
    "artificial intelligence",
    "machine learning",
    "data science",
    "software",
    "programmer",
    "developer",
    "engineer",
    "tech",
    "it ",
    "information technology",
]

# ============================================================================
# DEMOGRAPHICS
# ============================================================================

DEMOGRAPHIC_COLUMNS = {
    "age": "What is your age?",
    "gender": "What is your gender?",
    "education": "What is your highest achieved level of education?",
    "cs_expertise": "What is your level of expertise in Computer Science / AI?",
    "nlp_experience": "Please rate your experience with NLP",
    "job": JOB_COLUMN,
}

# Age groups mapped to ordinal numbers for regression
AGE_GROUP_MAPPING = {
    "18-24": 1,
    "25-34": 2,
    "35-44": 3,
    "45-54": 4,
    "55+": 5,
    "55-64": 5,
    "65+": 6,
}

# ============================================================================
# TUM COLOR PALETTE (plots)
# ============================================================================

TUM_BLUE = "#0065BD"         # Primary blue
TUM_BLUE_DARK = "#005293"    # Secondary dark blue
TUM_BLUE_DARKER = "#003359"  # Secondary darker blue
TUM_ORANGE = "#E37222"       # Accent orange
TUM_GREEN = "#A2AD00"        # Accent green
TUM_LIGHT_BLUE = "#98C6EA"   # Accent light blue
TUM_MED_BLUE = "#64A0C8"     # Accent medium blue
TUM_BEIGE = "#DAD7CB"        # Accent beige
TUM_GRAY_80 = "#333333"      # 80% black
TUM_GRAY_50 = "#808080"      # 50% black
TUM_GRAY_20 = "#CCCCCC"      # 20% black
TUM_BLACK = "#000000"
TUM_WHITE = "#FFFFFF"
