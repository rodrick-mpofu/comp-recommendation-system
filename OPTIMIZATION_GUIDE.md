# Model Optimization & Hyperparameter Tuning Guide

This guide explains how to optimize and tune the comp recommendation models for better performance.

## Overview

The system provides three optimization tools:

1. **Feature Importance Analysis** - Identify which features matter most
2. **Hyperparameter Tuning** - Find optimal model parameters
3. **Cross-Validation** - Robust performance evaluation

## Feature Importance Analysis

### What It Does

Analyzes which property features are most important for comp selection using Random Forest classification.

### How to Run

```bash
python run.py features
```

### Example Output

```
TOP 15 MOST IMPORTANT FEATURES
============================================================
 1. gla_diff                  ########### 0.2259
 2. gla                       ########## 0.2067
 3. property_type             ##### 0.1056
 4. age_diff                  ##### 0.1023
 5. stories                   ##### 0.1008
 6. room_count                #### 0.0956
 7. full_baths                #### 0.0801
 8. bedrooms                  ## 0.0491
```

### Key Insights

The analysis shows:
- **GLA Difference** is most important - properties with similar square footage to the subject are preferred
- **Actual GLA** matters - certain size ranges may be more desirable
- **Property Type** is critical - must match subject type
- **Age Difference** matters - similarly aged properties are preferred
- **Comparison features** (gla_diff, age_diff, bed_diff) are very important

### Files Created

- `tuning_results/feature_importance.csv` - Full results with all features

## Hyperparameter Tuning

### What It Does

Uses k-fold cross-validation to find the best hyperparameters for each model:

- **KNN**: Number of neighbors, distance metric
- **Clustering**: Number of clusters
- **Hybrid**: Weight balance between KNN and clustering

### How to Run

**Quick Mode** (faster, smaller parameter grid):
```bash
python run.py tune --quick
```

**Full Mode** (comprehensive, takes longer):
```bash
python run.py tune
```

### Parameter Grids

**Quick Mode:**
- KNN: 2 neighbor values, 2 metrics = 4 combinations
- Clustering: 2 cluster values = 2 combinations
- Hybrid: 3 weight values = 3 combinations

**Full Mode:**
- KNN: 5 neighbor values, 3 metrics = 15 combinations
- Clustering: 6 cluster values = 6 combinations
- Hybrid: 7 weight values = 7 combinations

### Example Output

```
TUNING SUMMARY
============================================================
     Model                        Best Params  Best MAP
       KNN n_neighbors=10, metric=euclidean      0.00
Clustering               n_clusters=10            0.25
    Hybrid          KNN=0.40, Cluster=0.60        0.25
```

### Files Created

- `tuning_results/knn_tuning_results.json` - KNN detailed results
- `tuning_results/clustering_tuning_results.json` - Clustering detailed results
- `tuning_results/hybrid_tuning_results.json` - Hybrid detailed results
- `tuning_results/tuning_summary.csv` - Summary table
- `models/knn_recommender_tuned.pkl` - Optimized KNN model
- `models/clustering_recommender_tuned.pkl` - Optimized clustering model
- `models/hybrid_recommender_tuned.pkl` - Optimized hybrid model

## Understanding the Metrics

### MAP (Mean Average Precision)

- Range: 0.0 to 1.0 (higher is better)
- Measures how well the model ranks relevant items
- **0.25** means the model gets 25% of rankings correct on average
- **0.00** means no correct rankings (model needs improvement)

### What the Results Tell Us

From the example output:

1. **Clustering outperforms KNN** (0.25 vs 0.00)
   - Grouping properties by similarity works better than pure nearest neighbors
   - Properties in the same cluster share important characteristics

2. **Hybrid model matches clustering** (0.25)
   - Best weight: 40% KNN, 60% Clustering
   - Clustering component dominates the hybrid

3. **Optimal Parameters:**
   - KNN: 10 neighbors with Euclidean distance
   - Clustering: 10 clusters
   - Hybrid: Lean more toward clustering

## Using Tuned Models

After running `python run.py tune`, optimized models are saved with `_tuned` suffix:

```bash
# Use tuned models for predictions
# (Update model_name to use _tuned versions)
python run.py predict --model hybrid  # Will use hybrid_recommender_tuned.pkl if available
```

## Advanced: Manual Tuning

You can also run tuning programmatically:

```python
from models.hyperparameter_tuning import HyperparameterTuner
from utils.data_utils import load_appraisals_data

df = load_appraisals_data('data/appraisals_dataset.json')
tuner = HyperparameterTuner(df, n_folds=5)

# Custom parameter grid
custom_grid = {
    'n_neighbors': [8, 12, 16],
    'metric': ['manhattan'],
    'n_recommendations': [5, 10]
}

results = tuner.grid_search_knn(custom_grid)
print(f"Best params: {results['best_params']}")
print(f"Best score: {results['best_score']}")
```

## Optimization Tips

### 1. Start with Feature Analysis

Always run feature importance first to understand your data:
```bash
python run.py features
```

### 2. Use Quick Mode First

Test with quick mode to get baseline results:
```bash
python run.py tune --quick
```

### 3. Run Full Tuning

Once you understand the patterns, run full tuning:
```bash
python run.py tune
```

### 4. Iterate on Features

Based on feature importance:
- Remove unimportant features (saves computation)
- Engineer new features (e.g., price_per_sqft)
- Try different distance calculations

### 5. Monitor Cross-Validation

Look at the standard deviation:
- `MAP: 0.25 (+/- 0.10)` - Consistent performance
- `MAP: 0.25 (+/- 0.50)` - High variance, unstable model

## Performance Benchmarks

On the current dataset (88 appraisals):

| Operation | Quick Mode | Full Mode |
|-----------|------------|-----------|
| Feature Analysis | ~30 seconds | ~30 seconds |
| KNN Tuning | ~2 minutes | ~5 minutes |
| Clustering Tuning | ~1 minute | ~3 minutes |
| Hybrid Tuning | ~2 minutes | ~7 minutes |
| **Total** | **~5 minutes** | **~15 minutes** |

## Interpreting Results

### Good Performance
- MAP > 0.3 - Model is learning useful patterns
- Low standard deviation - Consistent across folds
- Clear best parameters - Not all parameters tied

### Needs Improvement
- MAP < 0.1 - Model struggling to identify comps
- High standard deviation - Unstable predictions
- All parameters similar - May need more data or features

## Next Steps After Tuning

1. **Update default models** - Replace base models with tuned versions
2. **Add more features** - Based on importance analysis
3. **Collect more data** - More appraisals improve training
4. **Try ensemble methods** - Combine multiple models
5. **Implement A/B testing** - Compare tuned vs base models

## Troubleshooting

### "No valid evaluations"

This happens when the comp matching logic can't find matches. Solutions:
- Check that comp addresses match property addresses
- Verify data quality in the JSON file
- Adjust matching logic to be more flexible

### Low MAP scores

- Add more discriminative features
- Try different distance metrics
- Increase number of training examples
- Check for data quality issues

### High computation time

- Use `--quick` mode for faster iteration
- Reduce `n_folds` (default: 5)
- Limit parameter grid size
- Run on subset of data first

## Files Reference

### Input
- `data/appraisals_dataset.json` - Training data

### Output
- `tuning_results/feature_importance.csv`
- `tuning_results/knn_tuning_results.json`
- `tuning_results/clustering_tuning_results.json`
- `tuning_results/hybrid_tuning_results.json`
- `tuning_results/tuning_summary.csv`
- `models/*_recommender_tuned.pkl`

### Code
- `src/models/hyperparameter_tuning.py` - Tuning framework
- `src/models/feature_analysis.py` - Feature importance
- `src/models/tune_models.py` - Main tuning script
- `run.py` - CLI interface

## Summary

The optimization tools help you:
1. Understand which features matter (**features**)
2. Find best model parameters (**tune**)
3. Validate performance robustly (**cross-validation**)

Start with `python run.py features`, then `python run.py tune --quick`, and iterate based on the results!
