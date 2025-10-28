"""
Explainability module for comp recommendations.

This module provides LLM-based explanations for why properties are
recommended as comparable properties.
"""
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.feature_engineering import extract_property_features


class CompExplainer:
    """
    Generates natural language explanations for comp recommendations.
    """

    def __init__(self, use_llm: bool = False, llm_api_key: Optional[str] = None):
        """
        Initialize explainer.

        Args:
            use_llm: Whether to use LLM for explanations (requires API key)
            llm_api_key: API key for LLM service (OpenAI, Anthropic, etc.)
        """
        self.use_llm = use_llm
        self.llm_api_key = llm_api_key

        if use_llm and not llm_api_key:
            print("Warning: LLM enabled but no API key provided. Using rule-based explanations.")
            self.use_llm = False

    def explain_recommendation(
        self,
        subject_property: Dict,
        recommended_property: Dict,
        score: float,
        rank: int
    ) -> str:
        """
        Generate explanation for why a property was recommended.

        Args:
            subject_property: Subject property dict
            recommended_property: Recommended property dict
            score: Recommendation score
            rank: Rank of recommendation

        Returns:
            Natural language explanation
        """
        if self.use_llm:
            return self._explain_with_llm(subject_property, recommended_property, score, rank)
        else:
            return self._explain_with_rules(subject_property, recommended_property, score, rank)

    def _explain_with_rules(
        self,
        subject: Dict,
        comp: Dict,
        score: float,
        rank: int
    ) -> str:
        """
        Generate rule-based explanation.

        Args:
            subject: Subject property
            comp: Comparable property
            score: Score
            rank: Rank

        Returns:
            Explanation text
        """
        # Extract features
        subject_features = extract_property_features(subject, subject)
        comp_features = extract_property_features(comp, subject)

        explanation_parts = []

        # Header
        comp_address = comp.get('address', 'Unknown')
        explanation_parts.append(f"Recommendation #{rank}: {comp_address}")
        explanation_parts.append(f"Recommendation Score: {score:.3f}\n")

        # Similarities
        similarities = []

        # GLA similarity
        subject_gla = subject_features.get('gla', 0)
        comp_gla = comp_features.get('gla', 0)
        if subject_gla > 0:
            gla_diff_pct = abs(comp_gla - subject_gla) / subject_gla * 100
            if gla_diff_pct < 10:
                similarities.append(f"Very similar size ({comp_gla} vs {subject_gla} sq ft, {gla_diff_pct:.1f}% difference)")
            elif gla_diff_pct < 20:
                similarities.append(f"Similar size ({comp_gla} vs {subject_gla} sq ft, {gla_diff_pct:.1f}% difference)")

        # Bedroom similarity
        subject_beds = subject_features.get('bedrooms', 0)
        comp_beds = comp_features.get('bedrooms', 0)
        if subject_beds == comp_beds and subject_beds > 0:
            similarities.append(f"Same number of bedrooms ({int(comp_beds)})")
        elif abs(subject_beds - comp_beds) == 1:
            similarities.append(f"Similar bedrooms ({int(comp_beds)} vs {int(subject_beds)})")

        # Property type
        subject_type = subject.get('property_sub_type', subject.get('prop_type', ''))
        comp_type = comp.get('property_sub_type', comp.get('prop_type', ''))
        if subject_type and comp_type:
            if subject_type.lower() == comp_type.lower():
                similarities.append(f"Same property type ({comp_type})")

        # Age similarity
        subject_age = subject_features.get('age', 0)
        comp_age = comp_features.get('age', 0)
        if subject_age > 0 and comp_age > 0:
            age_diff = abs(comp_age - subject_age)
            if age_diff < 5:
                similarities.append(f"Similar age ({int(comp_age)} vs {int(subject_age)} years)")

        # Stories
        subject_stories = subject_features.get('stories', 0)
        comp_stories = comp_features.get('stories', 0)
        if subject_stories == comp_stories and subject_stories > 0:
            similarities.append(f"Same number of stories ({int(comp_stories)})")

        # Distance
        distance_km = comp_features.get('distance_km', 0)
        if distance_km > 0:
            if distance_km < 0.5:
                similarities.append(f"Very close proximity ({distance_km:.2f} km)")
            elif distance_km < 2:
                similarities.append(f"Close proximity ({distance_km:.2f} km)")

        # Build explanation
        if similarities:
            explanation_parts.append("KEY SIMILARITIES:")
            for sim in similarities:
                explanation_parts.append(f"  + {sim}")

        # Differences (potential concerns)
        differences = []

        # GLA difference
        if subject_gla > 0:
            gla_diff_pct = abs(comp_gla - subject_gla) / subject_gla * 100
            if gla_diff_pct >= 20:
                direction = "larger" if comp_gla > subject_gla else "smaller"
                differences.append(f"Size difference: {gla_diff_pct:.1f}% {direction}")

        # Bedroom difference
        if abs(subject_beds - comp_beds) >= 2:
            differences.append(f"Bedroom count differs by {abs(int(subject_beds - comp_beds))}")

        # Age difference
        if subject_age > 0 and comp_age > 0:
            age_diff = abs(comp_age - subject_age)
            if age_diff >= 10:
                direction = "newer" if comp_age < subject_age else "older"
                differences.append(f"Age: {age_diff} years {direction}")

        if differences:
            explanation_parts.append("\nNOTABLE DIFFERENCES:")
            for diff in differences:
                explanation_parts.append(f"  - {diff}")

        # Price info
        comp_price = comp.get('sale_price', comp.get('close_price'))
        if comp_price:
            try:
                price_num = float(str(comp_price).replace(',', '').replace('$', ''))
                explanation_parts.append(f"\nSale Price: ${price_num:,.0f}")
            except:
                explanation_parts.append(f"\nSale Price: {comp_price}")

        # Overall assessment
        explanation_parts.append("\nOVERALL ASSESSMENT:")
        if score >= 0.7:
            explanation_parts.append("  Excellent comparable - highly similar across key metrics")
        elif score >= 0.5:
            explanation_parts.append("  Good comparable - strong similarities with minor differences")
        elif score >= 0.3:
            explanation_parts.append("  Acceptable comparable - moderate similarities")
        else:
            explanation_parts.append("  Fair comparable - some key differences to consider")

        return "\n".join(explanation_parts)

    def _explain_with_llm(
        self,
        subject: Dict,
        comp: Dict,
        score: float,
        rank: int
    ) -> str:
        """
        Generate LLM-based explanation (placeholder for future implementation).

        Args:
            subject: Subject property
            comp: Comparable property
            score: Score
            rank: Rank

        Returns:
            Explanation text
        """
        # This would integrate with OpenAI, Anthropic Claude, etc.
        # For now, fall back to rule-based
        return self._explain_with_rules(subject, comp, score, rank)

    def explain_comparison(
        self,
        subject_property: Dict,
        property_a: Dict,
        property_b: Dict,
        score_a: float,
        score_b: float
    ) -> str:
        """
        Explain why property A is better/worse than property B as a comp.

        Args:
            subject_property: Subject property
            property_a: First property
            property_b: Second property
            score_a: Score for property A
            score_b: Score for property B

        Returns:
            Comparative explanation
        """
        addr_a = property_a.get('address', 'Property A')
        addr_b = property_b.get('address', 'Property B')

        explanation = []
        explanation.append(f"COMPARISON: {addr_a} vs {addr_b}\n")

        # Extract features
        subject_features = extract_property_features(subject_property, subject_property)
        features_a = extract_property_features(property_a, subject_property)
        features_b = extract_property_features(property_b, subject_property)

        # Compare scores
        if score_a > score_b:
            better = addr_a
            worse = addr_b
            score_diff = (score_a - score_b) / score_b * 100 if score_b > 0 else 100
        else:
            better = addr_b
            worse = addr_a
            score_diff = (score_b - score_a) / score_a * 100 if score_a > 0 else 100

        explanation.append(f"{better} scores {score_diff:.1f}% higher\n")

        # Analyze why
        explanation.append("KEY FACTORS:")

        # GLA comparison
        gla_diff_a = abs(features_a['gla_diff'])
        gla_diff_b = abs(features_b['gla_diff'])
        if abs(gla_diff_a - gla_diff_b) > 100:
            if gla_diff_a < gla_diff_b:
                explanation.append(f"  • {addr_a} is closer in size (off by {gla_diff_a:.0f} vs {gla_diff_b:.0f} sq ft)")
            else:
                explanation.append(f"  • {addr_b} is closer in size (off by {gla_diff_b:.0f} vs {gla_diff_a:.0f} sq ft)")

        # Bedroom comparison
        bed_diff_a = abs(features_a['bed_diff'])
        bed_diff_b = abs(features_b['bed_diff'])
        if bed_diff_a != bed_diff_b:
            if bed_diff_a < bed_diff_b:
                explanation.append(f"  • {addr_a} has matching bedrooms")
            else:
                explanation.append(f"  • {addr_b} has matching bedrooms")

        # Age comparison
        age_diff_a = abs(features_a['age_diff'])
        age_diff_b = abs(features_b['age_diff'])
        if abs(age_diff_a - age_diff_b) > 5:
            if age_diff_a < age_diff_b:
                explanation.append(f"  • {addr_a} is closer in age")
            else:
                explanation.append(f"  • {addr_b} is closer in age")

        return "\n".join(explanation)

    def explain_why_not_selected(
        self,
        subject_property: Dict,
        property_dict: Dict,
        score: float
    ) -> str:
        """
        Explain why a property was NOT selected as a comp.

        Args:
            subject_property: Subject property
            property_dict: Property that wasn't selected
            score: Its score

        Returns:
            Explanation of why it wasn't selected
        """
        address = property_dict.get('address', 'Unknown')
        subject_features = extract_property_features(subject_property, subject_property)
        prop_features = extract_property_features(property_dict, subject_property)

        explanation = []
        explanation.append(f"WHY {address} WAS NOT SELECTED\n")
        explanation.append(f"Score: {score:.3f} (Below threshold)\n")

        issues = []

        # Check GLA difference
        gla_diff_pct = abs(prop_features['gla_diff']) / subject_features['gla'] * 100 if subject_features['gla'] > 0 else 0
        if gla_diff_pct > 30:
            direction = "larger" if prop_features['gla'] > subject_features['gla'] else "smaller"
            issues.append(f"Size mismatch: {gla_diff_pct:.0f}% {direction} than subject")

        # Check bedroom difference
        if abs(prop_features['bed_diff']) >= 2:
            issues.append(f"Bedroom count differs by {abs(int(prop_features['bed_diff']))}")

        # Check property type
        subject_type = subject_property.get('property_sub_type', '')
        prop_type = property_dict.get('property_sub_type', '')
        if subject_type and prop_type and subject_type.lower() != prop_type.lower():
            issues.append(f"Different property type ({prop_type} vs {subject_type})")

        # Check age
        if abs(prop_features['age_diff']) > 20:
            direction = "newer" if prop_features['age'] < subject_features['age'] else "older"
            issues.append(f"Age: {abs(int(prop_features['age_diff']))} years {direction}")

        if issues:
            explanation.append("MAIN ISSUES:")
            for issue in issues:
                explanation.append(f"  X {issue}")
        else:
            explanation.append("This property may be acceptable but scored lower than other options.")

        return "\n".join(explanation)

    def generate_summary(
        self,
        subject_property: Dict,
        recommendations: List[Tuple[Dict, float]]
    ) -> str:
        """
        Generate overall summary of recommendations.

        Args:
            subject_property: Subject property
            recommendations: List of (property, score) tuples

        Returns:
            Summary text
        """
        summary = []
        summary.append("="*60)
        summary.append("COMP RECOMMENDATION SUMMARY")
        summary.append("="*60)

        subject_addr = subject_property.get('address', 'Unknown')
        subject_gla = extract_property_features(subject_property, subject_property).get('gla', 0)
        subject_beds = extract_property_features(subject_property, subject_property).get('bedrooms', 0)

        summary.append(f"\nSubject Property: {subject_addr}")
        summary.append(f"  GLA: {subject_gla} sq ft")
        summary.append(f"  Bedrooms: {int(subject_beds)}")

        summary.append(f"\nTop {len(recommendations)} Recommendations:\n")

        for i, (prop, score) in enumerate(recommendations, 1):
            addr = prop.get('address', 'Unknown')
            features = extract_property_features(prop, subject_property)
            gla = features.get('gla', 0)
            beds = features.get('bedrooms', 0)

            summary.append(f"{i}. {addr}")
            summary.append(f"   Score: {score:.3f} | GLA: {gla} | Beds: {int(beds)}")

        summary.append("\n" + "="*60)

        return "\n".join(summary)
