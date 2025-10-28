"""
Feature engineering utilities for comp recommendation system.

This module provides functions to extract and transform features from
properties and subject properties for use in ML models.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import re


def extract_numeric(value: str, default: float = 0.0) -> float:
    """
    Extract numeric value from string.

    Args:
        value: String containing numeric value
        default: Default value if extraction fails

    Returns:
        Extracted numeric value
    """
    if pd.isna(value) or value is None:
        return default

    if isinstance(value, (int, float)):
        return float(value)

    # Remove common units and formatting
    value_str = str(value).replace(',', '').replace('$', '')

    # Extract first number found
    match = re.search(r'[-+]?\d*\.?\d+', value_str)
    if match:
        return float(match.group())

    return default


def parse_distance(distance_str: str) -> float:
    """
    Parse distance string to kilometers.

    Args:
        distance_str: Distance string (e.g., "0.15 KM", "150 M")

    Returns:
        Distance in kilometers
    """
    if pd.isna(distance_str) or distance_str is None:
        return 0.0

    dist_str = str(distance_str).upper()

    # Extract number
    num = extract_numeric(dist_str, 0.0)

    # Convert to km if in meters
    if 'M' in dist_str and 'KM' not in dist_str:
        return num / 1000.0

    return num


def encode_categorical(value: str, categories: List[str]) -> int:
    """
    Encode categorical value as integer.

    Args:
        value: Categorical value
        categories: List of known categories

    Returns:
        Index of category, or -1 if unknown
    """
    if pd.isna(value) or value is None:
        return -1

    value_str = str(value).strip().lower()

    for i, cat in enumerate(categories):
        if cat.lower() in value_str:
            return i

    return -1


def extract_property_features(property_dict: Dict, subject_dict: Dict = None) -> Dict:
    """
    Extract features from a property dictionary.

    Args:
        property_dict: Dictionary containing property information
        subject_dict: Optional subject property for comparison features

    Returns:
        Dictionary of extracted features
    """
    features = {}

    # Basic numeric features
    features['gla'] = extract_numeric(property_dict.get('gla', 0))
    features['bedrooms'] = extract_numeric(property_dict.get('bedrooms', 0))
    features['bed_count'] = extract_numeric(property_dict.get('bed_count', 0))
    features['room_count'] = extract_numeric(property_dict.get('room_count', 0))
    features['full_baths'] = extract_numeric(property_dict.get('full_baths', 0))
    features['half_baths'] = extract_numeric(property_dict.get('half_baths', 0))
    features['age'] = extract_numeric(property_dict.get('age', 0))
    features['dom'] = extract_numeric(property_dict.get('dom', 0))  # Days on market

    # Sale price if available
    features['sale_price'] = extract_numeric(property_dict.get('sale_price', 0))

    # Distance to subject (if available)
    features['distance_km'] = parse_distance(property_dict.get('distance_to_subject', '0'))

    # Categorical features (encoded as integers)
    condition_cats = ['poor', 'fair', 'average', 'good', 'superior', 'excellent']
    features['condition'] = encode_categorical(
        property_dict.get('condition', ''),
        condition_cats
    )

    location_cats = ['inferior', 'similar', 'superior']
    features['location_similarity'] = encode_categorical(
        property_dict.get('location_similarity', ''),
        location_cats
    )

    # Property type encoding
    prop_type_cats = ['detached', 'semi-detached', 'townhouse', 'condo', 'apartment']
    features['property_type'] = encode_categorical(
        property_dict.get('property_sub_type', property_dict.get('prop_type', '')),
        prop_type_cats
    )

    # Number of stories
    stories_str = str(property_dict.get('stories', property_dict.get('levels', '0')))
    features['stories'] = extract_numeric(stories_str, 1.0)

    # Comparison features with subject (if provided)
    if subject_dict is not None:
        subject_gla = extract_numeric(subject_dict.get('gla', 0))
        if subject_gla > 0:
            features['gla_diff'] = abs(features['gla'] - subject_gla)
            features['gla_ratio'] = features['gla'] / subject_gla if subject_gla > 0 else 1.0
        else:
            features['gla_diff'] = 0
            features['gla_ratio'] = 1.0

        subject_beds = extract_numeric(subject_dict.get('num_beds', subject_dict.get('bedrooms', 0)))
        features['bed_diff'] = abs(features['bedrooms'] - subject_beds)

        subject_age = extract_numeric(subject_dict.get('subject_age', subject_dict.get('age', 0)))
        features['age_diff'] = abs(features['age'] - subject_age)

    return features


def create_feature_matrix(properties: List[Dict], subject_dict: Dict = None) -> pd.DataFrame:
    """
    Create feature matrix from list of properties.

    Args:
        properties: List of property dictionaries
        subject_dict: Optional subject property for comparison

    Returns:
        DataFrame with extracted features
    """
    features_list = []

    for prop in properties:
        features = extract_property_features(prop, subject_dict)
        features_list.append(features)

    df = pd.DataFrame(features_list)

    # Fill NaN values with 0
    df = df.fillna(0)

    return df


def normalize_features(df: pd.DataFrame, feature_cols: List[str] = None) -> pd.DataFrame:
    """
    Normalize features to 0-1 range.

    Args:
        df: DataFrame with features
        feature_cols: List of columns to normalize (None = all numeric)

    Returns:
        DataFrame with normalized features
    """
    df_norm = df.copy()

    if feature_cols is None:
        feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    for col in feature_cols:
        min_val = df[col].min()
        max_val = df[col].max()

        if max_val > min_val:
            df_norm[col] = (df[col] - min_val) / (max_val - min_val)
        else:
            df_norm[col] = 0

    return df_norm


def get_important_features() -> List[str]:
    """
    Get list of most important features for comparison.

    Returns:
        List of feature names
    """
    return [
        'gla',
        'bedrooms',
        'bed_count',
        'room_count',
        'full_baths',
        'age',
        'distance_km',
        'condition',
        'location_similarity',
        'property_type',
        'stories',
        'gla_diff',
        'bed_diff',
        'age_diff'
    ]
