import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from utils.data_utils import (
    load_appraisals_data,
    standardize_column_names,
    check_duplicates,
    get_missing_values_summary
)

def clean_appraisals_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess the appraisals dataset.
    
    Args:
        df (pd.DataFrame): Raw appraisals data
        
    Returns:
        pd.DataFrame: Cleaned data
    """
    # Make a copy to avoid modifying the original
    df_clean = df.copy()
    
    # Standardize column names
    df_clean = standardize_column_names(df_clean)
    
    # Print initial data info
    print("\nInitial Data Info:")
    print(f"Number of rows: {len(df_clean)}")
    print(f"Number of columns: {len(df_clean.columns)}")
    
    # Check for duplicates (only on simple columns, not lists/dicts)
    # Get columns that don't contain lists or dicts
    simple_columns = []
    for col in df_clean.columns:
        if df_clean[col].apply(lambda x: isinstance(x, (list, dict))).sum() == 0:
            simple_columns.append(col)

    if simple_columns:
        duplicates = check_duplicates(df_clean, subset=simple_columns)
        print(f"\nNumber of duplicate rows (based on simple columns): {len(duplicates)}")
    else:
        print("\nNo simple columns found for duplicate checking")
    
    # Get missing values summary
    missing_summary = get_missing_values_summary(df_clean)
    print("\nMissing Values Summary:")
    print(missing_summary)
    
    # TODO: Add more cleaning steps based on data analysis
    # This will be expanded as we analyze the data structure
    
    return df_clean

def main():
    # Get the project root directory
    project_root = Path(__file__).parent.parent.parent
    
    # Define input and output paths
    input_path = project_root / "data" / "appraisals_dataset.json"
    output_path = project_root / "data" / "cleaned_appraisals.csv"
    
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load and clean the data
    print("Loading data...")
    df = load_appraisals_data(str(input_path))
    
    print("Cleaning data...")
    df_clean = clean_appraisals_data(df)
    
    # Save cleaned data
    print(f"\nSaving cleaned data to {output_path}")
    df_clean.to_csv(output_path, index=False)
    print("Done!")

if __name__ == "__main__":
    main() 