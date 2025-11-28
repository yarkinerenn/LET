# Analysis Scripts

This folder contains all analysis scripts for the thesis project. All scripts have been updated to work with the organized folder structure.

## Quick Start

**Always run scripts from this directory (`scripts/`):**

```bash
cd "/Users/yarkineren/thesisXNLP/explainable-nlp/result scripts/scripts"
python script_name.py
```

## Script Organization

### 1. Data Processing
- **`script.py`** - Main data processing script. Reads raw Prolific data and generates processed Excel files with metrics.
  - Input: `../data/Prolific-big-small.xlsx`, `../data/Prolific-small-big.xlsx`
  - Output: `../data/experiment_results_with_metrics.xlsx` (and filtered versions)

### 2. Main Statistical Analysis
- **`h_tests.py`** - Hypothesis testing with cluster-robust standard errors. Runs all 16 hypotheses and generates summary tables.
  - Input: `../data/experiment_results_with_metrics_byjob_csai_only.xlsx`
  - Outputs: 4 CSV files in `../data/` + 24 plots in `../plots/general/`
  
- **`plots.py`** - Plotting functions library (imported by other scripts)

### 3. Demographic Analyses
Each demographic variable has two scripts:

**Analysis Scripts** (generate plots and statistical tests):
- `age_analysis.py` - Age group analysis
- `gender_analysis.py` - Gender analysis
- `cs_expertise_analysis.py` - CS/AI expertise analysis
- `education_level_analysis.py` - Education level analysis
- `nlp_experience_analysis.py` - NLP experience analysis

### 4. Reporting
- **`demographics_table.py`** - Generates LaTeX-formatted demographics table

## Output Locations

When you run these scripts, outputs will be saved to:

```
../data/                    # CSV and Excel files
../plots/general/           # Main analysis plots (from h_tests.py)
../plots/demographic_analyses/
  ├── age_plots/           # Age analysis outputs
  ├── gender_plots/        # Gender analysis outputs
  ├── cs_expertise_plots/  # CS expertise outputs
  ├── education_level_plots/ # Education analysis outputs
  └── nlp_experience_plots/  # NLP experience outputs
```

## Typical Workflow

1. **Process raw data** (only needed if raw data changes):
   ```bash
   python script.py
   ```

2. **Run main hypothesis tests**:
   ```bash
   python h_tests.py
   ```

3. **Run demographic analyses** (optional):
   ```bash
   python age_analysis.py
   python gender_analysis.py
   python cs_expertise_analysis.py
   python education_level_analysis.py
   python nlp_experience_analysis.py
   ```

4. **Generate demographics table**:
   ```bash
   python demographics_table.py
   ```