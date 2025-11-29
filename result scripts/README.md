# Result Scripts - Analysis Pipeline

This directory contains all analysis scripts for the thesis project. The scripts have been reorganized for better maintainability and ease of use.

## Quick Start

### Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running Analyses

**Run all analyses (recommended):**
```bash
python main.py
```

**Run specific parts:**
```bash
# Data processing only
python main.py --data-only

# Hypothesis tests only
python main.py --hypothesis-only

# All demographic analyses
python main.py --demographics-only

# Specific demographic analysis
python main.py --demographic age
python main.py --demographic gender
python main.py --demographic cs_expertise
python main.py --demographic education
python main.py --demographic nlp_experience
```

## Directory Structure

```
result scripts/
├── main.py                    # Main entry point - runs all analyses
├── config.py                 # Centralized configuration (paths, settings)
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── scripts/
│   ├── data_processing/
│   │   └── process_data.py   # Processes raw Prolific data, computes metrics
│   ├── analysis/
│   │   ├── hypothesis_tests.py  # Main hypothesis testing (16 hypotheses)
│   │   └── demographics_table.py # Generates LaTeX demographics table
│   ├── demographic_analysis/
│   │   ├── age_analysis.py
│   │   ├── gender_analysis.py
│   │   ├── cs_expertise_analysis.py
│   │   ├── education_level_analysis.py
│   │   └── nlp_experience_analysis.py
│   └── utils/
│       └── plots.py           # Plotting utilities (shared functions)
├── data/                     # Input/output data files
└── plots/                    # Generated plots
    ├── general/              # Main analysis plots
    └── demographic_analyses/ # Demographic-specific plots
```

## Script Descriptions

### Data Processing

**`scripts/data_processing/process_data.py`**
- Reads raw data files from both Prolific and friends/family sources
- Combines all four data files (Prolific-big-small, Prolific-small-big, experiment, experiment2)
- Computes metrics: RAIR, RSR, accuracy (initial/final), confidence deltas
- Outputs: `data/experiment_results_with_metrics.xlsx`

### Main Analysis

**`scripts/analysis/hypothesis_tests.py`**
- Tests 16 hypotheses using cluster-robust standard errors
- Accounts for within-subject correlation (repeated measures design)
- Outputs:
  - CSV files: hypothesis summaries, participant aggregates, within-subjects comparisons
  - 24 plots in `plots/general/`

**`scripts/analysis/demographics_table.py`**
- Generates LaTeX-formatted demographics table
- Prints to console (can be redirected to file)

### Demographic Analyses

Each demographic analysis script:
- Performs statistical tests (regression with cluster-robust SEs)
- Generates visualizations (box plots, violin plots, scatter plots)
- Saves outputs to respective subdirectories in `plots/demographic_analyses/`

**Available analyses:**
- `age_analysis.py` - Age group analysis
- `gender_analysis.py` - Gender analysis
- `cs_expertise_analysis.py` - CS/AI expertise analysis
- `education_level_analysis.py` - Education level analysis
- `nlp_experience_analysis.py` - NLP experience analysis

## Data Files

The analysis uses four data files located in the `data/` directory:

- **`Prolific-big-small.xlsx`** - Data from Prolific participants (Form A: big LLM first, small LLM second)
- **`Prolific-small-big.xlsx`** - Data from Prolific participants (Form B: small LLM first, big LLM second)
- **`experiment.xlsx`** - Data from friends and family (Form A: big LLM first, small LLM second)
- **`experiment2.xlsx`** - Data from friends and family (Form B: small LLM first, big LLM second)

All four files are automatically combined during data processing to create the final dataset with computed metrics.

### The orginal excel files:
- https://docs.google.com/spreadsheets/d/1wsG2bLVdLrXmwiKRuivRkuIT2KHmxuDsONkkVcm1Zyk/edit?usp=sharing
- https://docs.google.com/spreadsheets/d/14ofCkEJi32FVzOzTMw9ciYIcgokp5b5ImntU4YjVkb0/edit?usp=sharing
- https://docs.google.com/spreadsheets/d/1L3nPxmtOnsDnyFAThGoK065SQbzf-ZQB0oTc6e62FrU/edit?usp=sharing
- https://docs.google.com/spreadsheets/d/1gJ0_oEZse9qUz0TPf-SaJuTEGs-wCo6YXJ_DJMVBWto/edit?usp=sharing


## Configuration

All paths and settings are centralized in `config.py`. Key settings:

- **Paths**: Data and plot directories (automatically resolved relative to script location)
- **Experiment config**: Number of trials, ground truth labels, AI predictions, faithfulness labels
- **Job filter**: Options for filtering participants by job field
- **Demographic columns**: Column names for demographic variables

To modify settings, edit `config.py` directly.

## Output Locations

### Data Files
- `data/experiment_results_with_metrics.xlsx` - Main processed data
- `data/hypothesis_summary_table.csv` - Hypothesis test results summary
- `data/participant_level_aggregates.csv` - Participant-level metrics
- `data/within_subjects_comparisons_*.csv` - Within-subjects comparison results

### Plots
- `plots/general/` - Main analysis plots (24 plots from hypothesis tests)
- `plots/demographic_analyses/age_plots/` - Age analysis outputs
- `plots/demographic_analyses/gender_plots/` - Gender analysis outputs
- `plots/demographic_analyses/cs_expertise_plots/` - CS expertise outputs
- `plots/demographic_analyses/education_level_plots/` - Education analysis outputs
- `plots/demographic_analyses/nlp_experience_plots/` - NLP experience outputs

## Typical Workflow

1. **Process raw data** (only needed if raw data changes):
   ```bash
   python main.py --data-only
   ```

2. **Run main hypothesis tests**:
   ```bash
   python main.py --hypothesis-only
   ```

3. **Run all analyses** (recommended):
   ```bash
   python main.py
   ```

4. **Run specific demographic analysis**:
   ```bash
   python main.py --demographic age
   ```

## Running Scripts Individually

Scripts can also be run individually if needed. Make sure to run from the `result scripts/` directory:

```bash
cd "result scripts"
python scripts/data_processing/process_data.py
python scripts/analysis/hypothesis_tests.py
python scripts/demographic_analysis/age_analysis.py
```

**Note**: Scripts use absolute paths from `config.py`, so they should work regardless of where you run them from.

## Dependencies

See `requirements.txt` for full list. Main dependencies:
- pandas, numpy - Data manipulation
- scipy, statsmodels - Statistical analysis
- matplotlib, seaborn - Visualization
- openpyxl - Excel file handling

## Troubleshooting

**Import errors**: Make sure you've installed all dependencies (`pip install -r requirements.txt`)

**File not found errors**: Check that data files exist in `data/` directory. Run data processing first if needed.

**Path errors**: Scripts use `config.py` for paths, which resolves paths relative to the config file location. This should work regardless of where scripts are run from.

## Notes

- All scripts use cluster-robust standard errors to account for within-subject correlation (repeated measures design)
- Demographic analyses always use unfiltered data (all participants)
- Data processing can optionally filter by job field (configured in `config.py`)

