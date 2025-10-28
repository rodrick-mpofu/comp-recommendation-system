#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Verification script to ensure the project is set up correctly.
Run this after installation to verify everything works.
"""
import sys
import os
from pathlib import Path

# Fix encoding for Windows console
if sys.platform == 'win32':
    os.system('chcp 65001 > nul')
    sys.stdout.reconfigure(encoding='utf-8')

def check_imports():
    """Verify all required packages can be imported."""
    print("Checking imports...")
    required_packages = [
        'pandas',
        'numpy',
        'sklearn',
        'matplotlib',
        'seaborn',
        'plotly',
        'jupyter'
    ]

    failed = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} - NOT FOUND")
            failed.append(package)

    return len(failed) == 0

def check_data_files():
    """Verify required data files exist."""
    print("\nChecking data files...")
    data_dir = Path(__file__).parent / 'data'

    required_files = [
        'appraisals_dataset.json',
    ]

    failed = []
    for filename in required_files:
        filepath = data_dir / filename
        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024 * 1024)
            print(f"  ✓ {filename} ({size_mb:.2f} MB)")
        else:
            print(f"  ✗ {filename} - NOT FOUND")
            failed.append(filename)

    return len(failed) == 0

def check_project_structure():
    """Verify project structure is correct."""
    print("\nChecking project structure...")
    root = Path(__file__).parent

    required_paths = [
        'src',
        'src/__init__.py',
        'src/data_cleaning',
        'src/data_cleaning/__init__.py',
        'src/data_cleaning/clean_appraisals.py',
        'src/utils',
        'src/utils/__init__.py',
        'src/utils/data_utils.py',
        'src/models',
        'src/models/__init__.py',
        'notebooks',
        'notebooks/data_exploration.ipynb',
        'run.py',
        'requirements.txt',
    ]

    failed = []
    for path_str in required_paths:
        path = root / path_str
        if path.exists():
            print(f"  ✓ {path_str}")
        else:
            print(f"  ✗ {path_str} - NOT FOUND")
            failed.append(path_str)

    return len(failed) == 0

def test_data_loading():
    """Test if data can be loaded properly."""
    print("\nTesting data loading...")
    try:
        sys.path.insert(0, str(Path(__file__).parent / 'src'))
        from utils.data_utils import load_appraisals_data

        data_path = Path(__file__).parent / 'data' / 'appraisals_dataset.json'
        df = load_appraisals_data(str(data_path))

        print(f"  ✓ Data loaded successfully")
        print(f"    - {len(df)} appraisals")
        print(f"    - {len(df.columns)} columns")
        return True
    except Exception as e:
        print(f"  ✗ Data loading failed: {e}")
        return False

def main():
    """Run all verification checks."""
    print("=" * 60)
    print("Comp Recommendation System - Setup Verification")
    print("=" * 60)

    results = {
        'Imports': check_imports(),
        'Data Files': check_data_files(),
        'Project Structure': check_project_structure(),
        'Data Loading': test_data_loading(),
    }

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    all_passed = True
    for check, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{check}: {status}")
        if not passed:
            all_passed = False

    print("=" * 60)

    if all_passed:
        print("\n🎉 All checks passed! Your project is ready to use.")
        print("\nQuick Start:")
        print("  1. python run.py clean")
        print("  2. jupyter notebook notebooks/data_exploration.ipynb")
        return 0
    else:
        print("\n⚠️  Some checks failed. Please review the errors above.")
        print("\nTo fix:")
        print("  - Install missing packages: pip install -r requirements.txt")
        print("  - Ensure you're in the project root directory")
        return 1

if __name__ == '__main__':
    sys.exit(main())
