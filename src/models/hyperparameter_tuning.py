"""
Hyperparameter tuning utilities for comp recommendation models.

This module provides tools for optimizing model hyperparameters using
grid search, random search, and cross-validation.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import itertools
from collections import defaultdict
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.recommender import KNNRecommender, ClusterRecommender, HybridRecommender
from models.evaluation import RecommenderEvaluator
from utils.data_utils import load_appraisals_data
from utils.feature_engineering import create_feature_matrix, extract_property_features


class HyperparameterTuner:
    """
    Hyperparameter tuning framework for recommendation models.
    """

    def __init__(self, df: pd.DataFrame, n_folds: int = 5, random_state: int = 42):
        """
        Initialize tuner.

        Args:
            df: Appraisals dataframe
            n_folds: Number of folds for cross-validation
            random_state: Random seed for reproducibility
        """
        self.df = df
        self.n_folds = n_folds
        self.random_state = random_state
        self.results = []

    def create_folds(self) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Create k-fold splits of the data.

        Returns:
            List of (train_df, val_df) tuples
        """
        df_shuffled = self.df.sample(frac=1, random_state=self.random_state).reset_index(drop=True)
        fold_size = len(df_shuffled) // self.n_folds
        folds = []

        for i in range(self.n_folds):
            val_start = i * fold_size
            val_end = (i + 1) * fold_size if i < self.n_folds - 1 else len(df_shuffled)

            val_df = df_shuffled.iloc[val_start:val_end]
            train_df = pd.concat([
                df_shuffled.iloc[:val_start],
                df_shuffled.iloc[val_end:]
            ]).reset_index(drop=True)

            folds.append((train_df, val_df))

        return folds

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare feature matrix from appraisals.

        Args:
            df: Appraisals dataframe

        Returns:
            Feature matrix
        """
        all_properties = []
        all_subjects = []

        for idx, row in df.iterrows():
            properties = row.get('properties', [])
            if isinstance(properties, list):
                all_properties.extend(properties)

                subject_dict = {}
                for col in df.columns:
                    if col.startswith('subject.'):
                        subject_dict[col.replace('subject.', '')] = row[col]
                all_subjects.extend([subject_dict] * len(properties))

        features_list = []
        for prop, subject in zip(all_properties, all_subjects):
            features = extract_property_features(prop, subject)
            features_list.append(features)

        feature_df = pd.DataFrame(features_list).fillna(0)
        return feature_df

    def evaluate_on_fold(
        self,
        model,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Train and evaluate model on a single fold.

        Args:
            model: Model instance to train
            train_df: Training data
            val_df: Validation data

        Returns:
            Dictionary of evaluation metrics
        """
        # Prepare training features
        X_train = self.prepare_features(train_df)

        # Train model
        model.fit(X_train)

        # Evaluate on validation set
        evaluator = RecommenderEvaluator(k_values=[3, 5, 10])
        all_recommended = []
        all_actual = []

        for idx, row in val_df.iterrows():
            subject_dict = {}
            for col in val_df.columns:
                if col.startswith('subject.'):
                    subject_dict[col.replace('subject.', '')] = row[col]

            properties = row.get('properties', [])
            comps = row.get('comps', [])

            if not isinstance(properties, list) or len(properties) == 0:
                continue

            try:
                recommendations = model.recommend(subject_dict, properties)
                recommended_indices = [idx for idx, score in recommendations]

                # Match comps to properties
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

            except Exception:
                continue

        if len(all_recommended) > 0:
            metrics = evaluator.evaluate_multiple(all_recommended, all_actual)
            return metrics
        else:
            return {}

    def grid_search_knn(
        self,
        param_grid: Dict[str, List[Any]] = None
    ) -> Dict[str, Any]:
        """
        Grid search for KNN hyperparameters.

        Args:
            param_grid: Dictionary of parameter lists to search

        Returns:
            Dictionary with best parameters and results
        """
        if param_grid is None:
            param_grid = {
                'n_neighbors': [5, 10, 15, 20],
                'metric': ['euclidean', 'manhattan', 'cosine'],
                'n_recommendations': [5]
            }

        print("\n" + "="*60)
        print("KNN HYPERPARAMETER TUNING")
        print("="*60)
        print(f"Parameter grid: {param_grid}")
        print(f"Cross-validation folds: {self.n_folds}")

        folds = self.create_folds()
        param_combinations = list(itertools.product(*param_grid.values()))
        param_names = list(param_grid.keys())

        best_score = -1
        best_params = None
        results = []

        for i, param_values in enumerate(param_combinations):
            params = dict(zip(param_names, param_values))
            print(f"\nTesting {i+1}/{len(param_combinations)}: {params}")

            fold_metrics = []

            for fold_idx, (train_df, val_df) in enumerate(folds):
                model = KNNRecommender(**params)
                metrics = self.evaluate_on_fold(model, train_df, val_df)

                if metrics:
                    fold_metrics.append(metrics.get('MAP', 0))

            if fold_metrics:
                avg_map = np.mean(fold_metrics)
                std_map = np.std(fold_metrics)

                print(f"  MAP: {avg_map:.4f} (+/- {std_map:.4f})")

                results.append({
                    'params': params,
                    'mean_MAP': avg_map,
                    'std_MAP': std_map,
                    'fold_scores': fold_metrics
                })

                if avg_map > best_score:
                    best_score = avg_map
                    best_params = params

        print("\n" + "="*60)
        print(f"BEST PARAMETERS: {best_params}")
        print(f"BEST MAP SCORE: {best_score:.4f}")
        print("="*60)

        return {
            'best_params': best_params,
            'best_score': best_score,
            'all_results': results
        }

    def grid_search_clustering(
        self,
        param_grid: Dict[str, List[Any]] = None
    ) -> Dict[str, Any]:
        """
        Grid search for clustering hyperparameters.

        Args:
            param_grid: Dictionary of parameter lists to search

        Returns:
            Dictionary with best parameters and results
        """
        if param_grid is None:
            param_grid = {
                'n_clusters': [5, 8, 10, 12, 15],
                'n_recommendations': [5]
            }

        print("\n" + "="*60)
        print("CLUSTERING HYPERPARAMETER TUNING")
        print("="*60)
        print(f"Parameter grid: {param_grid}")
        print(f"Cross-validation folds: {self.n_folds}")

        folds = self.create_folds()
        param_combinations = list(itertools.product(*param_grid.values()))
        param_names = list(param_grid.keys())

        best_score = -1
        best_params = None
        results = []

        for i, param_values in enumerate(param_combinations):
            params = dict(zip(param_names, param_values))
            print(f"\nTesting {i+1}/{len(param_combinations)}: {params}")

            fold_metrics = []

            for fold_idx, (train_df, val_df) in enumerate(folds):
                model = ClusterRecommender(**params)
                metrics = self.evaluate_on_fold(model, train_df, val_df)

                if metrics:
                    fold_metrics.append(metrics.get('MAP', 0))

            if fold_metrics:
                avg_map = np.mean(fold_metrics)
                std_map = np.std(fold_metrics)

                print(f"  MAP: {avg_map:.4f} (+/- {std_map:.4f})")

                results.append({
                    'params': params,
                    'mean_MAP': avg_map,
                    'std_MAP': std_map,
                    'fold_scores': fold_metrics
                })

                if avg_map > best_score:
                    best_score = avg_map
                    best_params = params

        print("\n" + "="*60)
        print(f"BEST PARAMETERS: {best_params}")
        print(f"BEST MAP SCORE: {best_score:.4f}")
        print("="*60)

        return {
            'best_params': best_params,
            'best_score': best_score,
            'all_results': results
        }

    def optimize_hybrid_weights(
        self,
        knn_params: Dict[str, Any] = None,
        cluster_params: Dict[str, Any] = None,
        weight_range: List[float] = None
    ) -> Dict[str, Any]:
        """
        Optimize hybrid model weights.

        Args:
            knn_params: KNN parameters
            cluster_params: Clustering parameters
            weight_range: List of weights to try for KNN (cluster gets 1-weight)

        Returns:
            Dictionary with best weights and results
        """
        if weight_range is None:
            weight_range = [0.3, 0.4, 0.5, 0.6, 0.7]

        print("\n" + "="*60)
        print("HYBRID MODEL WEIGHT OPTIMIZATION")
        print("="*60)
        print(f"Weight range (KNN): {weight_range}")
        print(f"Cross-validation folds: {self.n_folds}")

        folds = self.create_folds()
        best_score = -1
        best_weight = None
        results = []

        for knn_weight in weight_range:
            cluster_weight = 1 - knn_weight
            print(f"\nTesting weights: KNN={knn_weight:.2f}, Cluster={cluster_weight:.2f}")

            fold_metrics = []

            for fold_idx, (train_df, val_df) in enumerate(folds):
                params = {
                    'n_recommendations': 5,
                    'knn_weight': knn_weight,
                    'cluster_weight': cluster_weight
                }
                model = HybridRecommender(**params)
                metrics = self.evaluate_on_fold(model, train_df, val_df)

                if metrics:
                    fold_metrics.append(metrics.get('MAP', 0))

            if fold_metrics:
                avg_map = np.mean(fold_metrics)
                std_map = np.std(fold_metrics)

                print(f"  MAP: {avg_map:.4f} (+/- {std_map:.4f})")

                results.append({
                    'knn_weight': knn_weight,
                    'cluster_weight': cluster_weight,
                    'mean_MAP': avg_map,
                    'std_MAP': std_map,
                    'fold_scores': fold_metrics
                })

                if avg_map > best_score:
                    best_score = avg_map
                    best_weight = knn_weight

        print("\n" + "="*60)
        print(f"BEST WEIGHTS: KNN={best_weight:.2f}, Cluster={1-best_weight:.2f}")
        print(f"BEST MAP SCORE: {best_score:.4f}")
        print("="*60)

        return {
            'best_knn_weight': best_weight,
            'best_cluster_weight': 1 - best_weight,
            'best_score': best_score,
            'all_results': results
        }

    def save_results(self, results: Dict[str, Any], filepath: str):
        """
        Save tuning results to JSON file.

        Args:
            results: Results dictionary
            filepath: Path to save file
        """
        # Convert numpy types to Python types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            return obj

        results_converted = convert_types(results)

        with open(filepath, 'w') as f:
            json.dump(results_converted, f, indent=2)

        print(f"\nResults saved to {filepath}")


def create_tuning_summary(
    knn_results: Dict[str, Any],
    clustering_results: Dict[str, Any],
    hybrid_results: Dict[str, Any]
) -> pd.DataFrame:
    """
    Create summary table of tuning results.

    Args:
        knn_results: KNN tuning results
        clustering_results: Clustering tuning results
        hybrid_results: Hybrid tuning results

    Returns:
        Summary DataFrame
    """
    summary_data = []

    # KNN
    if knn_results:
        summary_data.append({
            'Model': 'KNN',
            'Best Params': str(knn_results.get('best_params', {})),
            'Best MAP': knn_results.get('best_score', 0)
        })

    # Clustering
    if clustering_results:
        summary_data.append({
            'Model': 'Clustering',
            'Best Params': str(clustering_results.get('best_params', {})),
            'Best MAP': clustering_results.get('best_score', 0)
        })

    # Hybrid
    if hybrid_results:
        weights = f"KNN={hybrid_results.get('best_knn_weight', 0):.2f}, Cluster={hybrid_results.get('best_cluster_weight', 0):.2f}"
        summary_data.append({
            'Model': 'Hybrid',
            'Best Params': weights,
            'Best MAP': hybrid_results.get('best_score', 0)
        })

    return pd.DataFrame(summary_data)
