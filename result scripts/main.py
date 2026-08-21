#!/usr/bin/env python3
"""
Entry point for the analysis pipeline.

Usage:
    python main.py                     # Run everything
    python main.py --data-only         # Process raw data only
    python main.py --hypothesis-only   # Run the hypothesis tests only
    python main.py --demographics-only # Run all demographic analyses
    python main.py --demographic age   # Run one demographic analysis
"""

import argparse
import importlib
import sys
import traceback
from pathlib import Path

# Allow "python main.py" from any working directory
sys.path.insert(0, str(Path(__file__).resolve().parent))

from scripts import config

REQUIRED_PACKAGES = ["pandas", "numpy", "matplotlib", "seaborn", "scipy", "statsmodels", "openpyxl"]

# Pipeline steps, in the order they run
STEPS = {
    "data": ("DATA PROCESSING", "scripts.data_processing.process_data"),
    "hypothesis": ("HYPOTHESIS TESTS", "scripts.analysis.hypothesis_tests"),
    "demographics_table": ("DEMOGRAPHICS TABLE", "scripts.analysis.demographics_table"),
}

DEMOGRAPHIC_STEPS = {
    "age": "scripts.demographic_analysis.age_analysis",
    "gender": "scripts.demographic_analysis.gender_analysis",
    "cs_expertise": "scripts.demographic_analysis.cs_expertise_analysis",
    "education": "scripts.demographic_analysis.education_level_analysis",
    "nlp_experience": "scripts.demographic_analysis.nlp_experience_analysis",
}


def check_dependencies():
    """Return True if every required third-party package is importable."""
    missing = []
    for package in REQUIRED_PACKAGES:
        try:
            importlib.import_module(package)
        except ImportError:
            missing.append(package)

    if missing:
        print(f"ERROR: Missing required packages: {', '.join(missing)}")
        print("Please install them using: pip install -r requirements.txt")
        return False
    return True


def run_step(title, module_name):
    """Import a pipeline module and run its main(). Returns True on success."""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

    try:
        importlib.import_module(module_name).main()
    except Exception as exc:
        print(f"\n✗ Error during {title.lower()}: {exc}")
        traceback.print_exc()
        return False

    print(f"\n✓ {title.capitalize()} completed successfully")
    return True


def run_demographic_analysis(demographic):
    """Run a single demographic analysis."""
    title = f"{demographic.upper().replace('_', ' ')} ANALYSIS"
    return run_step(title, DEMOGRAPHIC_STEPS[demographic])


def run_all_demographics():
    """Run every demographic analysis and print a summary."""
    results = [(demo, run_demographic_analysis(demo)) for demo in DEMOGRAPHIC_STEPS]

    print("\n" + "=" * 60)
    print("DEMOGRAPHIC ANALYSES SUMMARY")
    print("=" * 60)
    for demo, success in results:
        print(f"{'✓' if success else '✗'} {demo}")

    return all(success for _, success in results)


def run_everything():
    """Run the full pipeline: data, hypotheses, demographics table, demographics."""
    print("\n" + "=" * 60)
    print("RUNNING ALL ANALYSES")
    print("=" * 60)

    if not run_step(*STEPS["data"]):
        print("\n✗ Data processing failed. Stopping.")
        sys.exit(1)

    success = run_step(*STEPS["hypothesis"])
    if not success:
        print("\n✗ Hypothesis tests failed. Continuing with demographics...")

    run_step(*STEPS["demographics_table"])
    success = run_all_demographics() and success

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    if success:
        print("✓ All analyses completed successfully!")
    else:
        print("⚠ Some analyses had errors. Check output above for details.")

    return success


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run analysis scripts for thesis project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                          # Run all analyses
  python main.py --data-only              # Process data only
  python main.py --hypothesis-only        # Run hypothesis tests only
  python main.py --demographics-only      # Run all demographic analyses
  python main.py --demographic age        # Run age analysis only
        """,
    )
    parser.add_argument("--data-only", action="store_true", help="Run data processing only")
    parser.add_argument("--hypothesis-only", action="store_true", help="Run hypothesis tests only")
    parser.add_argument("--demographics-only", action="store_true", help="Run all demographic analyses only")
    parser.add_argument("--demographic", choices=list(DEMOGRAPHIC_STEPS), help="Run specific demographic analysis")
    parser.add_argument("--skip-deps-check", action="store_true", help="Skip dependency check (not recommended)")
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.skip_deps_check:
        print("Checking dependencies...")
        if not check_dependencies():
            sys.exit(1)
        print("✓ All dependencies satisfied")

    config.ensure_output_dirs()

    if args.data_only:
        success = run_step(*STEPS["data"])
    elif args.hypothesis_only:
        success = run_step(*STEPS["hypothesis"])
    elif args.demographics_only:
        success = run_all_demographics()
    elif args.demographic:
        success = run_demographic_analysis(args.demographic)
    else:
        success = run_everything()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
