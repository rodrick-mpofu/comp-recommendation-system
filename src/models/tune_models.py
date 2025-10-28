"""
Main script for hyperparameter tuning.

Run this script to optimize model hyperparameters using cross-validation.
"""
import sys
from pathlib import Path
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.data_utils import load_appraisals_data
from models.hyperparameter_tuning import HyperparameterTuner, create_tuning_summary
from models.recommender import KNNRecommender, ClusterRecommender, HybridRecommender


def tune_all_models(
    data_path: str,
    n_folds: int = 5,
    quick: bool = False,
    output_dir: str = "tuning_results"
):
    """
    Tune all models and save results.

    Args:
        data_path: Path to appraisals dataset
        n_folds: Number of cross-validation folds
        quick: If True, use smaller parameter grids for faster tuning
        output_dir: Directory to save results
    """
    print("="*80)
    print("HYPERPARAMETER TUNING FOR COMP RECOMMENDATION MODELS")
    print("="*80)

    # Load data
    print(f"\nLoading data from {data_path}...")
    df = load_appraisals_data(data_path)
    print(f"Loaded {len(df)} appraisals")

    # Initialize tuner
    tuner = HyperparameterTuner(df, n_folds=n_folds)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    results = {}

    # 1. Tune KNN
    print("\n" + "="*80)
    print("STEP 1: TUNING KNN MODEL")
    print("="*80)

    if quick:
        knn_param_grid = {
            'n_neighbors': [10, 15],
            'metric': ['euclidean', 'manhattan'],
            'n_recommendations': [5]
        }
    else:
        knn_param_grid = {
            'n_neighbors': [5, 10, 15, 20, 25],
            'metric': ['euclidean', 'manhattan', 'cosine'],
            'n_recommendations': [5]
        }

    knn_results = tuner.grid_search_knn(knn_param_grid)
    results['knn'] = knn_results
    tuner.save_results(knn_results, str(output_path / 'knn_tuning_results.json'))

    # 2. Tune Clustering
    print("\n" + "="*80)
    print("STEP 2: TUNING CLUSTERING MODEL")
    print("="*80)

    if quick:
        clustering_param_grid = {
            'n_clusters': [8, 10],
            'n_recommendations': [5]
        }
    else:
        clustering_param_grid = {
            'n_clusters': [5, 8, 10, 12, 15, 20],
            'n_recommendations': [5]
        }

    clustering_results = tuner.grid_search_clustering(clustering_param_grid)
    results['clustering'] = clustering_results
    tuner.save_results(clustering_results, str(output_path / 'clustering_tuning_results.json'))

    # 3. Optimize Hybrid weights
    print("\n" + "="*80)
    print("STEP 3: OPTIMIZING HYBRID MODEL WEIGHTS")
    print("="*80)

    if quick:
        weight_range = [0.4, 0.5, 0.6]
    else:
        weight_range = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    hybrid_results = tuner.optimize_hybrid_weights(
        knn_params=knn_results['best_params'],
        cluster_params=clustering_results['best_params'],
        weight_range=weight_range
    )
    results['hybrid'] = hybrid_results
    tuner.save_results(hybrid_results, str(output_path / 'hybrid_tuning_results.json'))

    # 4. Create summary
    print("\n" + "="*80)
    print("TUNING SUMMARY")
    print("="*80)

    summary_df = create_tuning_summary(knn_results, clustering_results, hybrid_results)
    print("\n" + summary_df.to_string(index=False))

    summary_path = output_path / 'tuning_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary saved to {summary_path}")

    # 5. Train and save best models
    print("\n" + "="*80)
    print("TRAINING FINAL MODELS WITH BEST PARAMETERS")
    print("="*80)

    from models.train import prepare_training_data

    # Prepare all training data
    X_train, _ = prepare_training_data(df)

    # Train best models
    print("\nTraining KNN with best params...")
    best_knn = KNNRecommender(**knn_results['best_params'])
    best_knn.fit(X_train)

    print("Training Clustering with best params...")
    best_clustering = ClusterRecommender(**clustering_results['best_params'])
    best_clustering.fit(X_train)

    print("Training Hybrid with best params...")
    best_hybrid = HybridRecommender(
        n_recommendations=5,
        knn_weight=hybrid_results['best_knn_weight'],
        cluster_weight=hybrid_results['best_cluster_weight']
    )
    best_hybrid.fit(X_train)

    # Save models
    models_dir = Path(__file__).parent.parent.parent / "models"
    models_dir.mkdir(exist_ok=True)

    print(f"\nSaving optimized models to {models_dir}...")
    best_knn.save(str(models_dir / "knn_recommender_tuned.pkl"))
    best_clustering.save(str(models_dir / "clustering_recommender_tuned.pkl"))
    best_hybrid.save(str(models_dir / "hybrid_recommender_tuned.pkl"))

    print("\n" + "="*80)
    print("HYPERPARAMETER TUNING COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {output_path}")
    print(f"Optimized models saved to: {models_dir}")
    print("\nBest Parameters:")
    print(f"  KNN: {knn_results['best_params']} (MAP: {knn_results['best_score']:.4f})")
    print(f"  Clustering: {clustering_results['best_params']} (MAP: {clustering_results['best_score']:.4f})")
    print(f"  Hybrid: KNN weight={hybrid_results['best_knn_weight']:.2f} (MAP: {hybrid_results['best_score']:.4f})")

    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Hyperparameter Tuning for Comp Recommendation Models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/models/tune_models.py                    # Full tuning (takes time)
  python src/models/tune_models.py --quick             # Quick tuning with smaller grid
  python src/models/tune_models.py --folds 3           # Use 3-fold CV
  python src/models/tune_models.py --output results/   # Custom output directory
        """
    )

    parser.add_argument(
        '--data-path',
        type=str,
        default='data/appraisals_dataset.json',
        help='Path to appraisals dataset'
    )
    parser.add_argument(
        '--folds',
        type=int,
        default=5,
        help='Number of cross-validation folds (default: 5)'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Use smaller parameter grids for faster tuning'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='tuning_results',
        help='Output directory for results (default: tuning_results)'
    )

    args = parser.parse_args()

    # Run tuning
    tune_all_models(
        data_path=args.data_path,
        n_folds=args.folds,
        quick=args.quick,
        output_dir=args.output
    )


if __name__ == "__main__":
    main()
