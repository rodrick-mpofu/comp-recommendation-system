"""
Recommendation models for comparable properties.

This module implements various approaches to recommend comps:
- K-Nearest Neighbors based on feature similarity
- Clustering-based recommendations
- Hybrid scoring system
"""
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Tuple, Optional
import joblib
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils.feature_engineering import (
    extract_property_features,
    create_feature_matrix,
    normalize_features,
    get_important_features
)


class CompRecommender:
    """
    Base class for comparable property recommendation.
    """

    def __init__(self, n_recommendations: int = 5):
        """
        Initialize recommender.

        Args:
            n_recommendations: Number of comps to recommend
        """
        self.n_recommendations = n_recommendations
        self.scaler = StandardScaler()
        self.feature_cols = get_important_features()
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y: Optional[np.ndarray] = None):
        """
        Fit the recommender model.

        Args:
            X: Feature matrix
            y: Optional target labels (which properties were selected)
        """
        raise NotImplementedError("Subclasses must implement fit()")

    def recommend(self, subject_features: Dict, available_properties: List[Dict]) -> List[Tuple[int, float]]:
        """
        Recommend comparable properties.

        Args:
            subject_features: Subject property features
            available_properties: List of available property dicts

        Returns:
            List of (property_index, score) tuples, sorted by score
        """
        raise NotImplementedError("Subclasses must implement recommend()")

    def save(self, filepath: str):
        """Save model to disk."""
        joblib.dump(self, filepath)

    @staticmethod
    def load(filepath: str):
        """Load model from disk."""
        return joblib.load(filepath)


class KNNRecommender(CompRecommender):
    """
    K-Nearest Neighbors based recommender.

    Finds properties most similar to the subject property based on
    feature distance.
    """

    def __init__(self, n_recommendations: int = 5, n_neighbors: int = 10, metric: str = 'euclidean'):
        """
        Initialize KNN recommender.

        Args:
            n_recommendations: Number of comps to recommend
            n_neighbors: Number of neighbors to consider
            metric: Distance metric ('euclidean', 'manhattan', 'cosine')
        """
        super().__init__(n_recommendations)
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.knn = NearestNeighbors(n_neighbors=n_neighbors, metric=metric)

    def fit(self, X: pd.DataFrame, y: Optional[np.ndarray] = None):
        """
        Fit the KNN model.

        Args:
            X: Feature matrix of properties
            y: Not used for KNN
        """
        # Select and scale features
        X_subset = X[self.feature_cols].copy()
        X_scaled = self.scaler.fit_transform(X_subset)

        # Fit KNN
        self.knn.fit(X_scaled)
        self.is_fitted = True

    def recommend(self, subject_features: Dict, available_properties: List[Dict]) -> List[Tuple[int, float]]:
        """
        Recommend properties similar to subject.

        Args:
            subject_features: Subject property features dict
            available_properties: List of available property dicts

        Returns:
            List of (property_index, similarity_score) tuples
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making recommendations")

        # Create feature matrix for available properties
        props_df = create_feature_matrix(available_properties, subject_features)

        # Extract and scale subject features
        subject_df = pd.DataFrame([extract_property_features(subject_features, subject_features)])

        # Ensure all required features exist
        for col in self.feature_cols:
            if col not in subject_df.columns:
                subject_df[col] = 0
            if col not in props_df.columns:
                props_df[col] = 0

        subject_scaled = self.scaler.transform(subject_df[self.feature_cols])
        props_scaled = self.scaler.transform(props_df[self.feature_cols])

        # Find nearest neighbors
        distances, indices = self.knn.kneighbors(subject_scaled, n_neighbors=min(self.n_neighbors, len(props_scaled)))

        # Convert distances to similarity scores (closer = higher score)
        max_dist = distances[0].max() if len(distances[0]) > 0 else 1.0
        similarities = 1 - (distances[0] / (max_dist + 1e-6))

        # Create recommendations
        recommendations = [(indices[0][i], similarities[i]) for i in range(len(indices[0]))]

        # Sort by score (descending) and return top N
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:self.n_recommendations]


class ClusterRecommender(CompRecommender):
    """
    Clustering-based recommender.

    Groups properties into clusters and recommends from the same cluster
    as the subject property.
    """

    def __init__(self, n_recommendations: int = 5, n_clusters: int = 10):
        """
        Initialize cluster recommender.

        Args:
            n_recommendations: Number of comps to recommend
            n_clusters: Number of clusters to create
        """
        super().__init__(n_recommendations)
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)

    def fit(self, X: pd.DataFrame, y: Optional[np.ndarray] = None):
        """
        Fit clustering model.

        Args:
            X: Feature matrix
            y: Not used for clustering
        """
        # Select and scale features
        X_subset = X[self.feature_cols].copy()
        X_scaled = self.scaler.fit_transform(X_subset)

        # Fit clustering
        self.kmeans.fit(X_scaled)
        self.is_fitted = True

    def recommend(self, subject_features: Dict, available_properties: List[Dict]) -> List[Tuple[int, float]]:
        """
        Recommend properties from same cluster as subject.

        Args:
            subject_features: Subject property features
            available_properties: List of available properties

        Returns:
            List of (property_index, score) tuples
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making recommendations")

        # Create feature matrix
        props_df = create_feature_matrix(available_properties, subject_features)
        subject_df = pd.DataFrame([extract_property_features(subject_features, subject_features)])

        # Ensure all features exist
        for col in self.feature_cols:
            if col not in subject_df.columns:
                subject_df[col] = 0
            if col not in props_df.columns:
                props_df[col] = 0

        # Scale features
        subject_scaled = self.scaler.transform(subject_df[self.feature_cols])
        props_scaled = self.scaler.transform(props_df[self.feature_cols])

        # Predict cluster for subject
        subject_cluster = self.kmeans.predict(subject_scaled)[0]

        # Predict clusters for all properties
        prop_clusters = self.kmeans.predict(props_scaled)

        # Score based on cluster membership and distance to cluster center
        cluster_center = self.kmeans.cluster_centers_[subject_cluster]
        distances = np.linalg.norm(props_scaled - cluster_center, axis=1)

        # Properties in same cluster get higher base score
        scores = np.where(prop_clusters == subject_cluster, 1.0, 0.3)

        # Adjust by distance to cluster center (closer = higher)
        max_dist = distances.max() if len(distances) > 0 else 1.0
        distance_scores = 1 - (distances / (max_dist + 1e-6))
        scores = scores * distance_scores

        # Create recommendations
        recommendations = [(i, scores[i]) for i in range(len(scores))]

        # Sort and return top N
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:self.n_recommendations]


class HybridRecommender(CompRecommender):
    """
    Hybrid recommender combining KNN and clustering.

    Uses multiple strategies and combines their scores.
    """

    def __init__(self, n_recommendations: int = 5, knn_weight: float = 0.6, cluster_weight: float = 0.4):
        """
        Initialize hybrid recommender.

        Args:
            n_recommendations: Number of recommendations
            knn_weight: Weight for KNN scores
            cluster_weight: Weight for cluster scores
        """
        super().__init__(n_recommendations)
        self.knn_recommender = KNNRecommender(n_recommendations=n_recommendations)
        self.cluster_recommender = ClusterRecommender(n_recommendations=n_recommendations)
        self.knn_weight = knn_weight
        self.cluster_weight = cluster_weight

    def fit(self, X: pd.DataFrame, y: Optional[np.ndarray] = None):
        """Fit both models."""
        self.knn_recommender.fit(X, y)
        self.cluster_recommender.fit(X, y)
        self.is_fitted = True

    def recommend(self, subject_features: Dict, available_properties: List[Dict]) -> List[Tuple[int, float]]:
        """
        Recommend using hybrid approach.

        Args:
            subject_features: Subject property features
            available_properties: Available properties

        Returns:
            List of (property_index, combined_score) tuples
        """
        # Get recommendations from both models
        knn_recs = self.knn_recommender.recommend(subject_features, available_properties)
        cluster_recs = self.cluster_recommender.recommend(subject_features, available_properties)

        # Create score dictionaries
        knn_scores = {idx: score for idx, score in knn_recs}
        cluster_scores = {idx: score for idx, score in cluster_recs}

        # Combine scores for all properties
        all_indices = set(knn_scores.keys()) | set(cluster_scores.keys())
        combined_scores = {}

        for idx in all_indices:
            knn_score = knn_scores.get(idx, 0.0)
            cluster_score = cluster_scores.get(idx, 0.0)
            combined_scores[idx] = (self.knn_weight * knn_score +
                                   self.cluster_weight * cluster_score)

        # Sort and return top N
        recommendations = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        return recommendations[:self.n_recommendations]
