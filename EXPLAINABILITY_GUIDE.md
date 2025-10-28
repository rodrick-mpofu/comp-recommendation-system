# Explainability Guide - Understanding Comp Recommendations

This guide explains how to use the explainability features to understand why properties are recommended as comparables.

## Overview

The explainability module (`CompExplainer`) provides natural language explanations for comp recommendations, helping appraisers understand:
- **Why** a property was recommended
- **How** it compares to the subject property
- **What** makes it a good or poor comparable
- **Key similarities** and differences

## Quick Start

### Basic Usage

```bash
# Get predictions WITH explanations
python run.py predict --explain

# Predict for specific appraisal with explanations
python run.py predict --index 5 --explain
```

### Example Output

```
1. Unit 206 - 835 Milford Drive
   Score: 0.3421
   GLA: 950 | Beds: 2 | Price: $315,000

Recommendation #1: Unit 206 - 835 Milford Drive
Recommendation Score: 0.342

KEY SIMILARITIES:
  + Very similar size (950.0 vs 1044.0 sq ft, 9.0% difference)
  + Same property type (Townhouse)

NOTABLE DIFFERENCES:
  - Bedroom count differs by 2
  - Age: 15 years older

Sale Price: $315,000

OVERALL ASSESSMENT:
  Acceptable comparable - moderate similarities
```

## What Gets Explained

### 1. Key Similarities

The explainer identifies important similarities between the recommended property and the subject:

**Size (GLA)**
- "Very similar size" - Within 10% difference
- "Similar size" - Within 10-20% difference
- Shows exact square footage and percentage difference

**Bedrooms**
- "Same number of bedrooms" - Exact match
- "Similar bedrooms" - Differs by 1

**Property Type**
- "Same property type" - Detached, Townhouse, Condo, etc. match
- Critical for accurate comparisons

**Age**
- "Similar age" - Within 5 years
- Shows actual ages for reference

**Stories**
- "Same number of stories" - 1-story, 2-story, etc. match

**Proximity** (when available)
- "Very close proximity" - < 0.5 km
- "Close proximity" - < 2 km

### 2. Notable Differences

Highlights significant differences that may affect comparability:

**Size Mismatch**
- Flags differences >= 20%
- Indicates if property is larger or smaller

**Bedroom Difference**
- Flags differences of 2+ bedrooms
- Important for family size matching

**Age Difference**
- Flags differences of 10+ years
- Indicates if newer or older

### 3. Price Information

Shows the sale price (when available) to help assess value alignment.

### 4. Overall Assessment

Provides a summary rating based on the recommendation score:

| Score Range | Assessment |
|-------------|------------|
| >= 0.7 | Excellent comparable - highly similar |
| 0.5 - 0.7 | Good comparable - strong similarities |
| 0.3 - 0.5 | Acceptable comparable - moderate similarities |
| < 0.3 | Fair comparable - some key differences |

## Programmatic Usage

### Generate Explanations in Code

```python
from models.explainer import CompExplainer
from models.predict import predict_comps, load_model
from utils.data_utils import load_appraisals_data

# Load data and model
df = load_appraisals_data('data/appraisals_dataset.json')
model = load_model('models/hybrid_recommender.pkl')

# Get subject and properties
subject = {...}  # Subject property dict
properties = [...]  # List of available properties

# Make recommendations
recommendations = model.recommend(subject, properties)

# Create explainer
explainer = CompExplainer()

# Explain each recommendation
for rank, (prop_idx, score) in enumerate(recommendations, 1):
    prop = properties[prop_idx]
    explanation = explainer.explain_recommendation(
        subject, prop, score, rank
    )
    print(explanation)
```

### Compare Two Properties

```python
# Explain why property A is better than property B
comparison = explainer.explain_comparison(
    subject_property=subject,
    property_a=prop_a,
    property_b=prop_b,
    score_a=0.75,
    score_b=0.45
)
print(comparison)
```

Example output:
```
COMPARISON: 123 Main St vs 456 Oak Ave

123 Main St scores 66.7% higher

KEY FACTORS:
  - 123 Main St is closer in size (off by 50 vs 300 sq ft)
  - 123 Main St has matching bedrooms
  - 123 Main St is closer in age
```

### Explain Why NOT Selected

```python
# Explain why a property wasn't recommended
explanation = explainer.explain_why_not_selected(
    subject_property=subject,
    property_dict=rejected_prop,
    score=0.15
)
print(explanation)
```

Example output:
```
WHY 789 Pine St WAS NOT SELECTED

Score: 0.150 (Below threshold)

MAIN ISSUES:
  X Size mismatch: 45% larger than subject
  X Different property type (Condo vs Detached)
  X Age: 30 years older
```

### Generate Summary

```python
# Get overall summary of all recommendations
summary = explainer.generate_summary(
    subject_property=subject,
    recommendations=[(prop1, 0.8), (prop2, 0.7), (prop3, 0.6)]
)
print(summary)
```

## Explanation Logic

### How Similarities Are Determined

The explainer uses the following thresholds:

**GLA (Square Footage)**
- < 10% difference → "Very similar"
- 10-20% difference → "Similar"
- >= 20% difference → Flagged as notable difference

**Bedrooms**
- Exact match → "Same number"
- Differs by 1 → "Similar"
- Differs by 2+ → Flagged as notable difference

**Age**
- < 5 years difference → "Similar age"
- 10+ years → Flagged as notable difference

**Property Type**
- Must match for similarity (Detached, Townhouse, Condo)
- Mismatch flagged as issue

### Score Interpretation

The recommendation score (0-1) combines multiple factors:
- Feature similarity (GLA, beds, age, etc.)
- Distance to subject
- Property type match
- Overall compatibility

Higher scores indicate better comparables.

## Advanced Features

### Future: LLM Integration

The explainer is designed to support LLM-based explanations:

```python
# This will be available in future versions
explainer = CompExplainer(
    use_llm=True,
    llm_api_key="your-api-key"
)

# Will generate more sophisticated natural language explanations
explanation = explainer.explain_recommendation(subject, prop, score, rank)
```

LLM integration will provide:
- More nuanced language
- Context-aware explanations
- Market insights
- Comparative market analysis

### Custom Explanation Rules

You can extend the explainer with custom rules:

```python
class CustomExplainer(CompExplainer):
    def _explain_with_rules(self, subject, comp, score, rank):
        explanation = super()._explain_with_rules(subject, comp, score, rank)

        # Add custom analysis
        if comp.get('has_pool') and not subject.get('has_pool'):
            explanation += "\nNote: This property has a pool (not in subject)"

        return explanation
```

## Best Practices

### 1. Always Use Explanations for Final Recommendations

```bash
# For production use
python run.py predict --explain
```

Explanations help validate that recommendations make sense.

### 2. Compare Top Recommendations

When you have multiple good comps, use comparison:
```python
# Compare #1 vs #2
explainer.explain_comparison(subject, rec1, rec2, score1, score2)
```

### 3. Review Low-Scoring Recommendations

Use `explain_why_not_selected()` to understand what makes a property unsuitable.

### 4. Document Decision Rationale

Save explanations with your appraisal reports:
```python
with open('appraisal_report.txt', 'w') as f:
    f.write(explainer.generate_summary(subject, recommendations))
```

## Troubleshooting

### "No similarities found"

This means the property is quite different from the subject. Check:
- Property type mismatch?
- Large GLA difference?
- Very different age?

The "Notable Differences" section will show what's wrong.

### Unexpected recommendations

If a property scores high but looks wrong:
- Review the explanation to see what it's matching on
- Check if certain features are weighted too heavily
- Consider retraining with different hyperparameters

### Missing information

Some properties may lack certain fields (price, distance). The explainer handles this gracefully by showing "N/A" or omitting that comparison.

## Summary

The explainability system provides:
- **Transparency** - See why recommendations were made
- **Trust** - Understand the logic behind scores
- **Validation** - Verify recommendations make sense
- **Learning** - Understand what makes good comps

Use `--explain` flag for all important appraisals to ensure high-quality, understandable recommendations.

---

**Commands Reference:**

```bash
# Predictions with explanations
python run.py predict --explain

# Specific appraisal with explanations
python run.py predict --index N --explain

# Different model with explanations
python run.py predict --model knn --explain
```

For more details on the recommendation models themselves, see [README.md](README.md) and [OPTIMIZATION_GUIDE.md](OPTIMIZATION_GUIDE.md).
