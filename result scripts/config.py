"""
Centralized configuration for result scripts.

This module provides all paths, settings, and constants used across
the analysis scripts. Paths are resolved relative to this file's location,
making the scripts robust to where they are run from.
"""

from pathlib import Path
import os

# Get the directory containing this config file (result scripts/)
CONFIG_DIR = Path(__file__).parent.absolute()

# Directory paths (relative to config directory)
DATA_DIR = CONFIG_DIR / "data"
PLOTS_DIR = CONFIG_DIR / "plots"
SCRIPTS_DIR = CONFIG_DIR / "scripts"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)
SCRIPTS_DIR.mkdir(exist_ok=True)

# Data file paths
RAW_DATA_FILES = {
    "prolific_big_small": DATA_DIR / "Prolific-big-small.xlsx",
    "prolific_small_big": DATA_DIR / "Prolific-small-big.xlsx",
    "experiment": DATA_DIR / "experiment.xlsx",
    "experiment2": DATA_DIR / "experiment2.xlsx",
}

PROCESSED_DATA_FILE = DATA_DIR / "experiment_results_with_metrics.xlsx"
PROCESSED_DATA_FILE_CSAI_ONLY = DATA_DIR / "experiment_results_with_metrics_byjob_csai_only.xlsx"
PROCESSED_DATA_FILE_NO_CSAI = DATA_DIR / "experiment_results_with_metrics_byjob_no_csai.xlsx"

# Output file paths
OUTPUT_CSV_FILES = {
    "hypothesis_summary": DATA_DIR / "hypothesis_summary_table.csv",
    "participant_aggregates": DATA_DIR / "participant_level_aggregates.csv",
    "within_subjects_faithfulness": DATA_DIR / "within_subjects_comparisons_faithfulness.csv",
    "within_subjects_modelsize": DATA_DIR / "within_subjects_comparisons_modelsize.csv",
}

# Plot output directories
PLOT_DIRS = {
    "general": PLOTS_DIR / "general",
    "age": PLOTS_DIR / "demographic_analyses" / "age_plots",
    "gender": PLOTS_DIR / "demographic_analyses" / "gender_plots",
    "cs_expertise": PLOTS_DIR / "demographic_analyses" / "cs_expertise_plots",
    "education": PLOTS_DIR / "demographic_analyses" / "education_level_plots",
    "nlp_experience": PLOTS_DIR / "demographic_analyses" / "nlp_experience_plots",
}

# Ensure plot directories exist
for plot_dir in PLOT_DIRS.values():
    plot_dir.mkdir(parents=True, exist_ok=True)

# Experiment configuration
N_TRIALS = 16
BLOCK_SIZE = 5
START_COL_INDEX = 8  # 0-based index of the first trial column
SHEET_INDEX = 0

# Ground truth, AI predictions, and faithfulness labels
# Compact strings (no spaces), length should be n_trials
GT_STRING = "DTDTDTDTTDTDTDTD"   # Ground truth per question (D/T)
AI_STRING = "DTTDDTTDTDDTTDDT"   # AI prediction per question (D/T)
FAITH_STRING = "FUFUFUFUFUFUFUFU"   # Faithfulness per question (F/U)

# Convert to per-question lists
GT_LABELS = list(GT_STRING.strip())
AI_LABELS = list(AI_STRING.strip())
FAITH_LABELS = list(FAITH_STRING.strip())

# Job filter configuration
FILTER_BY_JOB = False
JOB_COLUMN = "What is your field of work/study ?"
FILTER_MODE = "include"  # "include" = only CS/AI, "exclude" = exclude CS/AI

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
    "information technology"
]

# Form assignments for data files
FILE_FORM_MAPPING = [
    (RAW_DATA_FILES["prolific_big_small"], "A"),
    (RAW_DATA_FILES["prolific_small_big"], "B"),
    (RAW_DATA_FILES["experiment"], "A"),
    (RAW_DATA_FILES["experiment2"], "B"),
]

# Demographic column names
DEMOGRAPHIC_COLUMNS = {
    "age": "What is your age?",
    "gender": "What is your gender?",
    "education": "What is your highest achieved level of education?",
    "cs_expertise": "What is your level of expertise in Computer Science / AI?",
    "nlp_experience": "Please rate your experience with NLP",
    "job": JOB_COLUMN,
}

# Age group mapping to ordinal numbers for regression
AGE_GROUP_MAPPING = {
    "18-24": 1,
    "25-34": 2,
    "35-44": 3,
    "45-54": 4,
    "55+": 5,
    "55-64": 5,
    "65+": 6
}

# TUM Color Palette (for plots)
TUM_BLUE = "#0065BD"
TUM_BLUE_DARK = "#005293"
TUM_BLUE_DARKER = "#003359"
TUM_ORANGE = "#E37222"
TUM_GREEN = "#A2AD00"
TUM_LIGHT_BLUE = "#98C6EA"
TUM_MED_BLUE = "#64A0C8"
TUM_BEIGE = "#DAD7CB"
TUM_GRAY_80 = "#333333"
TUM_GRAY_50 = "#808080"
TUM_GRAY_20 = "#CCCCCC"
TUM_BLACK = "#000000"
TUM_WHITE = "#FFFFFF"

