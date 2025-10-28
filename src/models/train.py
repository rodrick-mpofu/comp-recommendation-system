"""
Training script for comp recommendation models.

This script trains and evaluates different recommendation models
on the appraisal dataset.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import joblib

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.data_utils import load_appraisals_data
from utils.feature_engineering import create_feature_matrix, extract_property_features
from models.recommender import KNNRecommender, ClusterRecommender, HybridRecommender
from models.evaluation import RecommenderEvaluator, compare_models


def prepare_training_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, List]:
    """
    Prepare training data from appraisals dataframe.

    Args:
        df: Appraisals dataframe

    Returns:
        Tuple of (feature_matrix, comp_selections)
    """
    print("Preparing training data...")

    all_properties = []
    all_subjects = []

    # Collect all properties from all appraisals
    for idx, row in df.iterrows():
        properties = row.get('properties', [])
        if isinstance(properties, list):
            all_properties.extend(properties)

            # Add subject info to each property for context
            subject_dict = {}
            for col in df.columns:
                if col.startswith('subject.'):
                    subject_dict[col.replace('subject.', '')] = row[col]
            all_subjects.extend([subject_dict] * len(properties))

    print(f"Collected {len(all_properties)} properties from {len(df)} appraisals")

    # Create feature matrix
    # We'll create features with subject comparison
    features_list = []
    for prop, subject in zip(all_properties, all_subjects):
        features = extract_property_features(prop, subject)
        features_list.append(features)

    feature_df = pd.DataFrame(features_list)
    feature_df = feature_df.fillna(0)

    print(f"Created feature matrix: {feature_df.shape}")
    print(f"Features: {list(feature_df.columns)}")

    return feature_df, all_properties


def train_and_evaluate_models(df: pd.DataFrame, test_split: float = 0.2):
    """
    Train and evaluate all models.

    Args:
        df: Appraisals dataframe
        test_split: Fraction of data to use for testing

    Returns:
        Dictionary of trained models and results
    """
    print("\n" + "="*60)
    print("TRAINING AND EVALUATION")
    print("="*60)

    # Split data into train/test
    n_test = int(len(df) * test_split)
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df_train = df_shuffled[n_test:]
    df_test = df_shuffled[:n_test]

    print(f"\nDataset split:")
    print(f"  Training: {len(df_train)} appraisals")
    print(f"  Testing:  {len(df_test)} appraisals")

    # Prepare training features
    X_train, _ = prepare_training_data(df_train)

    # Initialize models
    models = {
        'KNN': KNNRecommender(n_recommendations=5, n_neighbors=10),
        'Clustering': ClusterRecommender(n_recommendations=5, n_clusters=8),
        'Hybrid': HybridRecommender(n_recommendations=5)
    }

    # Train models
    print("\nTraining models...")
    for name, model in models.items():
        print(f"  Training {name}...")
        model.fit(X_train)

    # Evaluate on test set
    print("\nEvaluating models on test set...")
    evaluator = RecommenderEvaluator(k_values=[3, 5, 10])
    results = {}

    for name, model in models.items():
        print(f"\nEvaluating {name}...")
        all_recommended = []
        all_actual = []

        for idx, row in df_test.iterrows():
            # Get subject features
            subject_dict = {}
            for col in df_test.columns:
                if col.startswith('subject.'):
                    subject_dict[col.replace('subject.', '')] = row[col]

            # Get available properties and actual selections
            properties = row.get('properties', [])
            comps = row.get('comps', [])

            if not isinstance(properties, list) or len(properties) == 0:
                continue

            # Make recommendations
            try:
                recommendations = model.recommend(subject_dict, properties)
                recommended_indices = [idx for idx, score in recommendations]

                # Find indices of actual selected comps
                # Match comps to properties by address or other unique identifier
                actual_indices = []
                for comp in comps:
                    if isinstance(comp, dict):
                        comp_addr = comp.get('address', '')
                        for pidx, prop in enumerate(properties):
                            if isinstance(prop, dict) and prop.get('address', '') == comp_addr:
                                actual_indices.append(pidx)
                                break

                if len(actual_indices) > 0:
                    all_recommended.append(recommended_indices)
                    all_actual.append(actual_indices)

            except Exception as e:
                print(f"  Error processing appraisal {idx}: {e}")
                continue

        # Calculate metrics
        if len(all_recommended) > 0:
            model_results = evaluator.evaluate_multiple(all_recommended, all_actual)
            results[name] = model_results
            evaluator.print_results(model_results)
        else:
            print(f"  No valid evaluations for {name}")

    # Compare models
    if len(results) > 1:
        print("\nMODEL COMPARISON:")
        comparison_df = compare_models(results)
        print(comparison_df.to_string())

    return models, results


def save_models(models: Dict, output_dir: str = "models"):
    """
    Save trained models to disk.

    Args:
        models: Dictionary of trained models
        output_dir: Directory to save models
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    print(f"\nSaving models to {output_path}...")
    for name, model in models.items():
        filepath = output_path / f"{name.lower()}_recommender.pkl"
        model.save(str(filepath))
        print(f"  Saved {name} to {filepath}")


def main():
    """Main training pipeline."""
    # Load data
    data_path = Path(__file__).parent.parent.parent / "data" / "appraisals_dataset.json"
    print(f"Loading data from {data_path}...")
    df = load_appraisals_data(str(data_path))
    print(f"Loaded {len(df)} appraisals\n")

    # Train and evaluate
    models, results = train_and_evaluate_models(df, test_split=0.2)

    # Save models
    models_dir = Path(__file__).parent.parent.parent / "models"
    save_models(models, str(models_dir))

    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"\nModels saved to: {models_dir}")
    print("\nYou can now use these models to make predictions with:")
    print("  python run.py predict")


if __name__ == "__main__":
    main()
