#!/usr/bin/env python
"""
Main entry point for the Comp Recommendation System

This script provides a simple CLI to run different parts of the system.
"""
import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data_cleaning.clean_appraisals import main as clean_data


def train_models():
    """Train recommendation models."""
    from models.train import main as train_main
    train_main()


def make_predictions(model_name='hybrid', appraisal_index=0):
    """Make predictions using trained models."""
    from models.predict import predict_for_appraisal, load_model
    from utils.data_utils import load_appraisals_data

    # Check if models exist
    models_dir = Path('models')
    if not models_dir.exists() or not list(models_dir.glob('*.pkl')):
        print("Error: No trained models found!")
        print("Please run 'python run.py train' first to train models.")
        return

    # Load data
    df = load_appraisals_data('data/appraisals_dataset.json')

    if appraisal_index >= len(df):
        print(f"Error: Appraisal index {appraisal_index} out of range (max: {len(df)-1})")
        return

    # Get appraisal
    appraisal = df.iloc[appraisal_index]

    # Extract subject
    subject = {}
    for col in df.columns:
        if col.startswith('subject.'):
            subject[col.replace('subject.', '')] = appraisal[col]

    # Get properties
    properties = appraisal.get('properties', [])

    if not isinstance(properties, list) or len(properties) == 0:
        print(f"Error: No properties found for appraisal {appraisal_index}")
        return

    # Select model
    model_file = f"{model_name}_recommender.pkl"
    model_path = models_dir / model_file

    if not model_path.exists():
        print(f"Error: Model '{model_name}' not found.")
        print(f"Available models:")
        for p in models_dir.glob('*.pkl'):
            print(f"  - {p.stem.replace('_recommender', '')}")
        return

    # Make predictions
    print(f"\nAppraisal #{appraisal_index}")
    print(f"Subject: {subject.get('address', 'Unknown')}")
    print(f"Available properties: {len(properties)}\n")

    predict_for_appraisal(
        str(model_path),
        subject,
        properties,
        n_recommendations=5,
        verbose=True
    )


def evaluate_models():
    """Evaluate trained models."""
    from models.train import train_and_evaluate_models
    from utils.data_utils import load_appraisals_data

    # Check if models exist
    models_dir = Path('models')
    if not models_dir.exists() or not list(models_dir.glob('*.pkl')):
        print("Error: No trained models found!")
        print("Please run 'python run.py train' first to train models.")
        return

    print("Evaluating all models on test set...")
    df = load_appraisals_data('data/appraisals_dataset.json')
    train_and_evaluate_models(df, test_split=0.3)


def main():
    parser = argparse.ArgumentParser(
        description="Comp Recommendation System - Main Entry Point",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py clean                    # Clean the dataset
  python run.py train                    # Train all models
  python run.py predict                  # Make predictions (default: hybrid model)
  python run.py predict --model knn      # Use KNN model
  python run.py predict --index 5        # Predict for appraisal #5
  python run.py evaluate                 # Evaluate all models
        """
    )
    parser.add_argument(
        'command',
        choices=['clean', 'train', 'predict', 'evaluate'],
        help='Command to execute'
    )
    parser.add_argument(
        '--data-path',
        type=str,
        default='data/appraisals_dataset.json',
        help='Path to the appraisals dataset'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='hybrid',
        choices=['knn', 'clustering', 'hybrid'],
        help='Model to use for prediction (default: hybrid)'
    )
    parser.add_argument(
        '--index',
        type=int,
        default=0,
        help='Appraisal index for prediction (default: 0)'
    )

    args = parser.parse_args()

    if args.command == 'clean':
        print("Running data cleaning pipeline...")
        clean_data()
        print("\nData cleaning completed successfully!")

    elif args.command == 'train':
        print("Training recommendation models...")
        print("This may take a few minutes...\n")
        train_models()

    elif args.command == 'predict':
        make_predictions(model_name=args.model, appraisal_index=args.index)

    elif args.command == 'evaluate':
        evaluate_models()


if __name__ == '__main__':
    main()
