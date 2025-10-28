import pandas as pd
import numpy as np
from typing import Dict, List, Union
import json

def load_appraisals_data(file_path: str) -> pd.DataFrame:
    """
    Load the appraisals dataset from JSON file.

    Args:
        file_path (str): Path to the JSON file

    Returns:
        pd.DataFrame: Loaded data as a pandas DataFrame
    """
    with open(file_path, 'r') as f:
        data = json.load(f)

    # Handle nested JSON structure where data is under 'appraisals' key
    if isinstance(data, dict) and 'appraisals' in data:
        data = data['appraisals']

    return pd.json_normalize(data)

def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names to lowercase with underscores.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        
    Returns:
        pd.DataFrame: DataFrame with standardized column names
    """
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    return df

def check_duplicates(df: pd.DataFrame, subset: List[str] = None) -> pd.DataFrame:
    """
    Check for duplicate rows in the DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        subset (List[str], optional): Columns to check for duplicates
        
    Returns:
        pd.DataFrame: DataFrame containing only duplicate rows
    """
    return df[df.duplicated(subset=subset, keep=False)]

def get_missing_values_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get a summary of missing values in the DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        
    Returns:
        pd.DataFrame: Summary of missing values
    """
    missing_values = df.isnull().sum()
    missing_percentage = (missing_values / len(df)) * 100
    return pd.DataFrame({
        'Missing Values': missing_values,
        'Percentage': missing_percentage
    }).sort_values('Missing Values', ascending=False) 