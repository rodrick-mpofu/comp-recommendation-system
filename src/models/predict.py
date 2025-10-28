"""
Prediction interface for comp recommendation.

This module provides functions to load trained models and make
predictions for new appraisals.
"""
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.recommender import CompRecommender
from utils.feature_engineering import extract_property_features


def load_model(model_path: str) -> CompRecommender:
    """
    Load a trained model from disk.

    Args:
        model_path: Path to saved model file

    Returns:
        Loaded model
    """
    return CompRecommender.load(model_path)


def predict_comps(
    model: CompRecommender,
    subject_property: Dict,
    available_properties: List[Dict],
    return_scores: bool = True
) -> List[Tuple[Dict, float]]:
    """
    Predict comparable properties for a subject property.

    Args:
        model: Trained recommender model
        subject_property: Subject property dictionary
        available_properties: List of available property dictionaries
        return_scores: Whether to return scores with properties

    Returns:
        List of (property_dict, score) tuples if return_scores=True,
        otherwise just list of property dicts
    """
    # Get recommendations
    recommendations = model.recommend(subject_property, available_properties)

    # Map indices back to property dictionaries
    results = []
    for idx, score in recommendations:
        if 0 <= idx < len(available_properties):
            prop = available_properties[idx]
            if return_scores:
                results.append((prop, score))
            else:
                results.append(prop)

    return results


def format_recommendation(
    property_dict: Dict,
    score: float,
    rank: int
) -> str:
    """
    Format a recommendation for display.

    Args:
        property_dict: Property dictionary
        score: Recommendation score
        rank: Rank (1-based)

    Returns:
        Formatted string
    """
    address = property_dict.get('address', 'Unknown')
    gla = property_dict.get('gla', 'N/A')
    bedrooms = property_dict.get('bedrooms', property_dict.get('bed_count', 'N/A'))

    # Handle both sale_price (comps) and close_price (properties)
    price = property_dict.get('sale_price', property_dict.get('close_price', 'N/A'))
    if price != 'N/A' and price not in [None, '']:
        # Format price nicely
        try:
            price_num = float(str(price).replace(',', '').replace('$', ''))
            price = f"${price_num:,.0f}"
        except:
            pass

    distance = property_dict.get('distance_to_subject', 'N/A')

    output = f"\n{rank}. {address}"
    output += f"\n   Score: {score:.4f}"
    output += f"\n   GLA: {gla} | Beds: {bedrooms} | Price: {price}"
    if distance != 'N/A':
        output += f" | Distance: {distance}"

    return output


def predict_for_appraisal(
    model_path: str,
    subject_property: Dict,
    available_properties: List[Dict],
    n_recommendations: int = 5,
    verbose: bool = True
) -> List[Tuple[Dict, float]]:
    """
    End-to-end prediction for an appraisal.

    Args:
        model_path: Path to trained model
        subject_property: Subject property dict
        available_properties: Available properties list
        n_recommendations: Number of recommendations to return
        verbose: Print recommendations

    Returns:
        List of (property, score) tuples
    """
    # Load model
    if verbose:
        print(f"Loading model from {model_path}...")
    model = load_model(model_path)

    # Make predictions
    if verbose:
        print(f"Making predictions for {len(available_properties)} available properties...")

    recommendations = predict_comps(
        model,
        subject_property,
        available_properties,
        return_scores=True
    )

    # Limit to requested number
    recommendations = recommendations[:n_recommendations]

    # Print results
    if verbose:
        print("\n" + "="*60)
        print("RECOMMENDED COMPARABLE PROPERTIES")
        print("="*60)

        for rank, (prop, score) in enumerate(recommendations, 1):
            print(format_recommendation(prop, score, rank))

        print("\n" + "="*60)

    return recommendations


def predict_from_json(
    model_path: str,
    appraisal_json: str,
    output_path: Optional[str] = None
) -> List[Dict]:
    """
    Make predictions from a JSON file containing appraisal data.

    Args:
        model_path: Path to trained model
        appraisal_json: Path to JSON file with appraisal data
        output_path: Optional path to save recommendations

    Returns:
        List of recommended properties
    """
    # Load appraisal data
    with open(appraisal_json, 'r') as f:
        appraisal_data = json.load(f)

    # Extract subject and properties
    subject = appraisal_data.get('subject', {})
    properties = appraisal_data.get('properties', [])

    # Make predictions
    recommendations = predict_for_appraisal(
        model_path,
        subject,
        properties,
        verbose=True
    )

    # Save if requested
    if output_path:
        output_data = {
            'subject': subject,
            'recommendations': [
                {
                    'rank': i + 1,
                    'property': prop,
                    'score': float(score)
                }
                for i, (prop, score) in enumerate(recommendations)
            ]
        }

        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"\nRecommendations saved to {output_path}")

    return [prop for prop, score in recommendations]


def main():
    """Demo prediction."""
    import sys
    from utils.data_utils import load_appraisals_data

    # Load sample data
    data_path = Path(__file__).parent.parent.parent / "data" / "appraisals_dataset.json"
    df = load_appraisals_data(str(data_path))

    # Use first appraisal as example
    first_appraisal = df.iloc[0]

    # Extract subject
    subject = {}
    for col in df.columns:
        if col.startswith('subject.'):
            subject[col.replace('subject.', '')] = first_appraisal[col]

    # Get properties
    properties = first_appraisal.get('properties', [])

    # Try to load and predict
    models_dir = Path(__file__).parent.parent.parent / "models"
    model_files = list(models_dir.glob("*.pkl"))

    if len(model_files) == 0:
        print("No trained models found. Please run 'python run.py train' first.")
        return

    # Use first available model
    model_path = model_files[0]
    print(f"Using model: {model_path.name}\n")

    # Make predictions
    predict_for_appraisal(
        str(model_path),
        subject,
        properties,
        n_recommendations=5,
        verbose=True
    )


if __name__ == "__main__":
    main()
