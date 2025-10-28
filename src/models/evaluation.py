"""
Evaluation metrics and utilities for comp recommendation models.

This module provides metrics to evaluate how well the models
match human appraiser selections.
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Set
from collections import defaultdict


def precision_at_k(recommended: List[int], actual: List[int], k: int = None) -> float:
    """
    Calculate precision@k: fraction of recommended items that are relevant.

    Args:
        recommended: List of recommended property indices
        actual: List of actual selected property indices
        k: Number of top recommendations to consider (None = all)

    Returns:
        Precision score (0-1)
    """
    if k is not None:
        recommended = recommended[:k]

    if len(recommended) == 0:
        return 0.0

    actual_set = set(actual)
    relevant_recommended = sum(1 for idx in recommended if idx in actual_set)

    return relevant_recommended / len(recommended)


def recall_at_k(recommended: List[int], actual: List[int], k: int = None) -> float:
    """
    Calculate recall@k: fraction of relevant items that are recommended.

    Args:
        recommended: List of recommended property indices
        actual: List of actual selected property indices
        k: Number of top recommendations to consider (None = all)

    Returns:
        Recall score (0-1)
    """
    if k is not None:
        recommended = recommended[:k]

    if len(actual) == 0:
        return 0.0

    actual_set = set(actual)
    relevant_recommended = sum(1 for idx in recommended if idx in actual_set)

    return relevant_recommended / len(actual)


def f1_score(precision: float, recall: float) -> float:
    """
    Calculate F1 score (harmonic mean of precision and recall).

    Args:
        precision: Precision score
        recall: Recall score

    Returns:
        F1 score
    """
    if precision + recall == 0:
        return 0.0

    return 2 * (precision * recall) / (precision + recall)


def average_precision(recommended: List[int], actual: List[int]) -> float:
    """
    Calculate Average Precision (AP).

    Args:
        recommended: List of recommended property indices (ordered)
        actual: List of actual selected property indices

    Returns:
        Average precision score
    """
    if len(actual) == 0:
        return 0.0

    actual_set = set(actual)
    precisions = []
    num_relevant = 0

    for k, idx in enumerate(recommended, 1):
        if idx in actual_set:
            num_relevant += 1
            precisions.append(num_relevant / k)

    if len(precisions) == 0:
        return 0.0

    return np.mean(precisions)


def mean_average_precision(all_recommended: List[List[int]], all_actual: List[List[int]]) -> float:
    """
    Calculate Mean Average Precision (MAP) across multiple appraisals.

    Args:
        all_recommended: List of recommendation lists
        all_actual: List of actual selection lists

    Returns:
        MAP score
    """
    aps = [average_precision(rec, act) for rec, act in zip(all_recommended, all_actual)]
    return np.mean(aps) if len(aps) > 0 else 0.0


def ndcg_at_k(recommended: List[int], actual: List[int], k: int = None) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain (NDCG@k).

    Args:
        recommended: List of recommended property indices (ordered)
        actual: List of actual selected property indices
        k: Number of top recommendations to consider

    Returns:
        NDCG score (0-1)
    """
    if k is not None:
        recommended = recommended[:k]

    if len(recommended) == 0 or len(actual) == 0:
        return 0.0

    actual_set = set(actual)

    # Calculate DCG
    dcg = 0.0
    for i, idx in enumerate(recommended, 1):
        if idx in actual_set:
            dcg += 1.0 / np.log2(i + 1)

    # Calculate ideal DCG
    ideal_length = min(len(actual), len(recommended))
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_length))

    return dcg / idcg if idcg > 0 else 0.0


def hit_rate(recommended: List[int], actual: List[int]) -> float:
    """
    Calculate hit rate: 1 if any recommended item is in actual, else 0.

    Args:
        recommended: List of recommended property indices
        actual: List of actual selected property indices

    Returns:
        Hit rate (0 or 1)
    """
    actual_set = set(actual)
    return 1.0 if any(idx in actual_set for idx in recommended) else 0.0


def mean_reciprocal_rank(all_recommended: List[List[int]], all_actual: List[List[int]]) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR).

    Args:
        all_recommended: List of recommendation lists
        all_actual: List of actual selection lists

    Returns:
        MRR score
    """
    reciprocal_ranks = []

    for recommended, actual in zip(all_recommended, all_actual):
        actual_set = set(actual)

        for rank, idx in enumerate(recommended, 1):
            if idx in actual_set:
                reciprocal_ranks.append(1.0 / rank)
                break
        else:
            reciprocal_ranks.append(0.0)

    return np.mean(reciprocal_ranks) if len(reciprocal_ranks) > 0 else 0.0


class RecommenderEvaluator:
    """
    Comprehensive evaluator for recommendation models.
    """

    def __init__(self, k_values: List[int] = [3, 5, 10]):
        """
        Initialize evaluator.

        Args:
            k_values: List of k values for precision@k and recall@k
        """
        self.k_values = k_values
        self.results = defaultdict(list)

    def evaluate_single(self, recommended: List[int], actual: List[int]) -> Dict[str, float]:
        """
        Evaluate a single set of recommendations.

        Args:
            recommended: Recommended property indices
            actual: Actual selected property indices

        Returns:
            Dictionary of metric scores
        """
        metrics = {}

        # Precision and recall at different k values
        for k in self.k_values:
            metrics[f'precision@{k}'] = precision_at_k(recommended, actual, k)
            metrics[f'recall@{k}'] = recall_at_k(recommended, actual, k)
            metrics[f'f1@{k}'] = f1_score(
                metrics[f'precision@{k}'],
                metrics[f'recall@{k}']
            )
            metrics[f'ndcg@{k}'] = ndcg_at_k(recommended, actual, k)

        # Overall metrics
        metrics['average_precision'] = average_precision(recommended, actual)
        metrics['hit_rate'] = hit_rate(recommended, actual)

        return metrics

    def evaluate_multiple(self, all_recommended: List[List[int]], all_actual: List[List[int]]) -> Dict[str, float]:
        """
        Evaluate multiple recommendations and return aggregated metrics.

        Args:
            all_recommended: List of recommendation lists
            all_actual: List of actual selection lists

        Returns:
            Dictionary of aggregated metric scores
        """
        # Collect individual metrics
        all_metrics = []
        for recommended, actual in zip(all_recommended, all_actual):
            metrics = self.evaluate_single(recommended, actual)
            all_metrics.append(metrics)

        # Aggregate
        aggregated = {}
        if len(all_metrics) > 0:
            for key in all_metrics[0].keys():
                values = [m[key] for m in all_metrics]
                aggregated[f'mean_{key}'] = np.mean(values)
                aggregated[f'std_{key}'] = np.std(values)

        # Add global metrics
        aggregated['MAP'] = mean_average_precision(all_recommended, all_actual)
        aggregated['MRR'] = mean_reciprocal_rank(all_recommended, all_actual)

        return aggregated

    def print_results(self, results: Dict[str, float]):
        """
        Pretty print evaluation results.

        Args:
            results: Dictionary of metric scores
        """
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)

        # Group by metric type
        for k in self.k_values:
            print(f"\nMetrics @ {k}:")
            print(f"  Precision: {results.get(f'mean_precision@{k}', 0):.4f}")
            print(f"  Recall:    {results.get(f'mean_recall@{k}', 0):.4f}")
            print(f"  F1:        {results.get(f'mean_f1@{k}', 0):.4f}")
            print(f"  NDCG:      {results.get(f'mean_ndcg@{k}', 0):.4f}")

        print(f"\nOverall Metrics:")
        print(f"  MAP:       {results.get('MAP', 0):.4f}")
        print(f"  MRR:       {results.get('MRR', 0):.4f}")
        print(f"  Hit Rate:  {results.get('mean_hit_rate', 0):.4f}")

        print("="*60 + "\n")


def compare_models(models_results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """
    Compare multiple models' results.

    Args:
        models_results: Dictionary mapping model names to their results

    Returns:
        DataFrame comparing models
    """
    comparison_df = pd.DataFrame(models_results).T

    # Select key metrics for comparison
    key_metrics = [col for col in comparison_df.columns
                   if col.startswith('mean_') and '@5' in col or
                   col in ['MAP', 'MRR']]

    if len(key_metrics) > 0:
        return comparison_df[key_metrics].sort_values('MAP', ascending=False)
    else:
        return comparison_df
