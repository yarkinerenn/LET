#!/usr/bin/env python3
"""
Main entry point for running all analysis scripts.

This script provides a command-line interface to run:
- Data processing
- Hypothesis tests
- Demographic analyses
- All analyses (default)

Usage:
    python main.py                    # Run all analyses
    python main.py --data-only        # Process data only
    python main.py --hypothesis-only  # Run hypothesis tests only
    python main.py --demographics-only # Run all demographic analyses
    python main.py --demographic age  # Run specific demographic analysis
"""

import argparse
import sys
import subprocess
from pathlib import Path
import importlib.util

# Add scripts directory to path
SCRIPTS_DIR = Path(__file__).parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

def check_dependencies():
    """Check if required packages are installed."""
    required_packages = [
        'pandas', 'numpy', 'matplotlib', 'seaborn', 
        'scipy', 'statsmodels', 'openpyxl'
    ]
    missing = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"ERROR: Missing required packages: {', '.join(missing)}")
        print("Please install them using: pip install -r requirements.txt")
        return False
    return True

def run_data_processing():
    """Run data processing script."""
    print("\n" + "="*60)
    print("STEP 1: DATA PROCESSING")
    print("="*60)
    
    try:
        # Import and run the data processing module
        spec = importlib.util.spec_from_file_location(
            "process_data", 
            SCRIPTS_DIR / "data_processing" / "process_data.py"
        )
        if spec is None or spec.loader is None:
            print("ERROR: Could not load process_data.py")
            return False
        
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Run the main function
        if hasattr(module, 'main'):
            module.main()
        elif hasattr(module, '__main__'):
            # Execute if __name__ == "__main__" block
            exec(open(SCRIPTS_DIR / "data_processing" / "process_data.py").read())
        else:
            print("ERROR: process_data.py does not have a main() function")
            return False
        
        print("\n✓ Data processing completed successfully")
        return True
    except Exception as e:
        print(f"\n✗ Error during data processing: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_hypothesis_tests():
    """Run hypothesis testing script."""
    print("\n" + "="*60)
    print("STEP 2: HYPOTHESIS TESTS")
    print("="*60)
    
    try:
        spec = importlib.util.spec_from_file_location(
            "hypothesis_tests",
            SCRIPTS_DIR / "analysis" / "hypothesis_tests.py"
        )
        if spec is None or spec.loader is None:
            print("ERROR: Could not load hypothesis_tests.py")
            return False
        
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        if hasattr(module, 'main'):
            module.main()
        else:
            exec(open(SCRIPTS_DIR / "analysis" / "hypothesis_tests.py").read())
        
        print("\n✓ Hypothesis tests completed successfully")
        return True
    except Exception as e:
        print(f"\n✗ Error during hypothesis tests: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_demographics_table():
    """Run demographics table generation."""
    print("\n" + "="*60)
    print("STEP 3: DEMOGRAPHICS TABLE")
    print("="*60)
    
    try:
        spec = importlib.util.spec_from_file_location(
            "demographics_table",
            SCRIPTS_DIR / "analysis" / "demographics_table.py"
        )
        if spec is None or spec.loader is None:
            print("ERROR: Could not load demographics_table.py")
            return False
        
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        if hasattr(module, 'main'):
            module.main()
        else:
            exec(open(SCRIPTS_DIR / "analysis" / "demographics_table.py").read())
        
        print("\n✓ Demographics table generated successfully")
        return True
    except Exception as e:
        print(f"\n✗ Error generating demographics table: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_demographic_analysis(demographic):
    """Run a specific demographic analysis."""
    demographic_scripts = {
        'age': 'age_analysis.py',
        'gender': 'gender_analysis.py',
        'cs_expertise': 'cs_expertise_analysis.py',
        'education': 'education_level_analysis.py',
        'nlp_experience': 'nlp_experience_analysis.py',
    }
    
    if demographic not in demographic_scripts:
        print(f"ERROR: Unknown demographic '{demographic}'")
        print(f"Available options: {', '.join(demographic_scripts.keys())}")
        return False
    
    script_name = demographic_scripts[demographic]
    script_path = SCRIPTS_DIR / "demographic_analysis" / script_name
    
    print(f"\n" + "="*60)
    print(f"RUNNING: {demographic.upper().replace('_', ' ')} ANALYSIS")
    print("="*60)
    
    try:
        spec = importlib.util.spec_from_file_location(
            demographic,
            script_path
        )
        if spec is None or spec.loader is None:
            print(f"ERROR: Could not load {script_name}")
            return False
        
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        if hasattr(module, 'main'):
            module.main()
        else:
            exec(open(script_path).read())
        
        print(f"\n✓ {demographic} analysis completed successfully")
        return True
    except Exception as e:
        print(f"\n✗ Error during {demographic} analysis: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_all_demographics():
    """Run all demographic analyses."""
    demographics = ['age', 'gender', 'cs_expertise', 'education', 'nlp_experience']
    results = []
    
    for demo in demographics:
        success = run_demographic_analysis(demo)
        results.append((demo, success))
    
    print("\n" + "="*60)
    print("DEMOGRAPHIC ANALYSES SUMMARY")
    print("="*60)
    for demo, success in results:
        status = "✓" if success else "✗"
        print(f"{status} {demo}")
    
    return all(success for _, success in results)

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run analysis scripts for thesis project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                          # Run all analyses
  python main.py --data-only               # Process data only
  python main.py --hypothesis-only         # Run hypothesis tests only
  python main.py --demographics-only       # Run all demographic analyses
  python main.py --demographic age        # Run age analysis only
        """
    )
    
    parser.add_argument(
        '--data-only',
        action='store_true',
        help='Run data processing only'
    )
    parser.add_argument(
        '--hypothesis-only',
        action='store_true',
        help='Run hypothesis tests only'
    )
    parser.add_argument(
        '--demographics-only',
        action='store_true',
        help='Run all demographic analyses only'
    )
    parser.add_argument(
        '--demographic',
        choices=['age', 'gender', 'cs_expertise', 'education', 'nlp_experience'],
        help='Run specific demographic analysis'
    )
    parser.add_argument(
        '--skip-deps-check',
        action='store_true',
        help='Skip dependency check (not recommended)'
    )
    
    args = parser.parse_args()
    
    # Check dependencies
    if not args.skip_deps_check:
        print("Checking dependencies...")
        if not check_dependencies():
            sys.exit(1)
        print("✓ All dependencies satisfied")
    
    # Determine what to run
    success = True
    
    if args.data_only:
        success = run_data_processing()
    elif args.hypothesis_only:
        success = run_hypothesis_tests()
    elif args.demographics_only:
        success = run_all_demographics()
    elif args.demographic:
        success = run_demographic_analysis(args.demographic)
    else:
        # Run everything
        print("\n" + "="*60)
        print("RUNNING ALL ANALYSES")
        print("="*60)
        
        # Step 1: Data processing
        if not run_data_processing():
            print("\n✗ Data processing failed. Stopping.")
            sys.exit(1)
        
        # Step 2: Hypothesis tests
        if not run_hypothesis_tests():
            print("\n✗ Hypothesis tests failed. Continuing with demographics...")
            success = False
        
        # Step 3: Demographics table
        run_demographics_table()
        
        # Step 4: Demographic analyses
        if not run_all_demographics():
            success = False
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)
        if success:
            print("✓ All analyses completed successfully!")
        else:
            print("⚠ Some analyses had errors. Check output above for details.")
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()

