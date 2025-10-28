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


def main():
    parser = argparse.ArgumentParser(
        description="Comp Recommendation System - Main Entry Point"
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

    args = parser.parse_args()

    if args.command == 'clean':
        print("Running data cleaning pipeline...")
        clean_data()
        print("\nData cleaning completed successfully!")

    elif args.command == 'train':
        print("Training models...")
        print("Note: Model training is not yet implemented.")
        print("This will be part of Milestone 1: Statistical Modeling")

    elif args.command == 'predict':
        print("Making predictions...")
        print("Note: Prediction is not yet implemented.")
        print("Models need to be trained first.")

    elif args.command == 'evaluate':
        print("Evaluating model performance...")
        print("Note: Evaluation is not yet implemented.")
        print("Models need to be trained first.")


if __name__ == '__main__':
    main()
