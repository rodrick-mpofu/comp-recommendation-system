"""
Feature importance analysis for comp recommendation.

This module analyzes which features are most important for
making good comp recommendations.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.data_utils import load_appraisals_data
from utils.feature_engineering import (
    extract_property_features,
    get_important_features
)


def create_training_dataset_for_importance(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Create labeled dataset for feature importance analysis.

    For each appraisal, properties that were selected as comps get label=1,
    others get label=0.

    Args:
        df: Appraisals dataframe

    Returns:
        Tuple of (features_df, labels)
    """
    all_features = []
    all_labels = []

    for idx, row in df.iterrows():
        properties = row.get('properties', [])
        comps = row.get('comps', [])

        if not isinstance(properties, list) or not isinstance(comps, list):
            continue

        # Get subject dict
        subject_dict = {}
        for col in df.columns:
            if col.startswith('subject.'):
                subject_dict[col.replace('subject.', '')] = row[col]

        # Get comp addresses
        comp_addresses = set()
        for comp in comps:
            if isinstance(comp, dict):
                comp_addresses.add(comp.get('address', ''))

        # Extract features and labels
        for prop in properties:
            if isinstance(prop, dict):
                features = extract_property_features(prop, subject_dict)
                all_features.append(features)

                # Label: 1 if this property was selected as comp, 0 otherwise
                is_comp = prop.get('address', '') in comp_addresses
                all_labels.append(1 if is_comp else 0)

    features_df = pd.DataFrame(all_features).fillna(0)
    labels = np.array(all_labels)

    return features_df, labels


def analyze_feature_importance(
    df: pd.DataFrame,
    n_estimators: int = 100,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Analyze feature importance using Random Forest.

    Args:
        df: Appraisals dataframe
        n_estimators: Number of trees in random forest
        random_state: Random seed

    Returns:
        DataFrame with feature importances
    """
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*60)

    # Create labeled dataset
    print("\nCreating labeled dataset...")
    X, y = create_training_dataset_for_importance(df)

    print(f"Dataset: {len(X)} properties")
    print(f"Selected as comps: {y.sum()} ({y.mean()*100:.1f}%)")
    print(f"Features: {len(X.columns)}")

    # Train Random Forest
    print("\nTraining Random Forest classifier...")
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        max_depth=10,
        n_jobs=-1
    )

    # Use only important features
    feature_cols = get_important_features()
    available_cols = [col for col in feature_cols if col in X.columns]

    X_subset = X[available_cols]
    rf.fit(X_subset, y)

    # Get feature importances
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]

    # Create results dataframe
    results = pd.DataFrame({
        'Feature': [available_cols[i] for i in indices],
        'Importance': importances[indices],
        'Rank': range(1, len(importances) + 1)
    })

    return results


def print_feature_importance_report(importance_df: pd.DataFrame, top_n: int = 15):
    """
    Print feature importance report.

    Args:
        importance_df: DataFrame with feature importances
        top_n: Number of top features to display
    """
    print("\n" + "="*60)
    print(f"TOP {top_n} MOST IMPORTANT FEATURES")
    print("="*60)

    for idx, row in importance_df.head(top_n).iterrows():
        bar_length = int(row['Importance'] * 50)
        bar = '#' * bar_length
        print(f"{row['Rank']:2d}. {row['Feature']:25s} {bar} {row['Importance']:.4f}")

    print("\n" + "="*60)
    print("FEATURE IMPORTANCE INSIGHTS")
    print("="*60)

    # Get top features
    top_features = importance_df.head(5)['Feature'].tolist()

    print("\nTop 5 most important features for comp selection:")
    for i, feature in enumerate(top_features, 1):
        print(f"  {i}. {feature}")

    # Analyze feature categories
    comparison_features = [f for f in top_features if 'diff' in f or 'ratio' in f]
    if comparison_features:
        print(f"\nComparison features are important: {', '.join(comparison_features)}")
        print("This suggests properties similar to the subject are preferred.")

    distance_features = [f for f in top_features if 'distance' in f]
    if distance_features:
        print(f"\nDistance is important: {', '.join(distance_features)}")
        print("This suggests proximity to subject matters.")

    property_features = [f for f in top_features if f in ['gla', 'bedrooms', 'age', 'condition']]
    if property_features:
        print(f"\nIntrinsic property features are important: {', '.join(property_features)}")
        print("This suggests certain property characteristics are key indicators.")


def main():
    """Run feature importance analysis."""
    # Load data
    data_path = Path(__file__).parent.parent.parent / "data" / "appraisals_dataset.json"
    print(f"Loading data from {data_path}...")
    df = load_appraisals_data(str(data_path))
    print(f"Loaded {len(df)} appraisals\n")

    # Analyze importance
    importance_df = analyze_feature_importance(df, n_estimators=100)

    # Print report
    print_feature_importance_report(importance_df, top_n=15)

    # Save results
    output_path = Path("tuning_results")
    output_path.mkdir(exist_ok=True)

    csv_path = output_path / "feature_importance.csv"
    importance_df.to_csv(csv_path, index=False)
    print(f"\nFeature importance saved to {csv_path}")


if __name__ == "__main__":
    main()
