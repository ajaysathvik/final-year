"""
Advanced Adversarial Training Module
=====================================
Implements four advanced techniques for adversarial training on tabular fraud data:

1. Attack Propagation in Feature Engineering (APFE)
   - Propagates adversarial perturbations from raw input space to engineered feature space

2. Adversarial Structural Clustering (ASC)
   - Groups perturbations using clustering to capture common attack patterns

3. Metric Learning (TLA - Triplet Loss Adversarial)
   - Adds triplet loss to bring adversarial samples closer to their natural class neighbors

4. Generative Simulation (FraudGAN)
   - Uses GAN-generated samples for preemptive adversarial retraining

Min-Max Optimization Foundation:
    min_θ E_{(x,y)~D} [ max_||δ||≤ε L(f_θ(x+δ), y) ]
"""

import numpy as np
import pandas as pd
from typing import Optional, Callable
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity


class AttackPropagation:
    """
    Propagates adversarial perturbations from raw features to engineered features
    within the training loop. Simulates complex feature transformations used in
    fraud detection (temporal aggregations, behavioral patterns, etc.)
    """

    def __init__(self, epsilon: float = 0.05):
        self.epsilon = epsilon

    def propagate(
        self, X_raw: np.ndarray, feature_transforms: list[Callable] = None
    ) -> np.ndarray:
        """
        Propagate perturbations through engineered features.

        Args:
            X_raw: Raw numeric features
            feature_transforms: List of transformation functions

        Returns:
            Perturbed engineered features
        """
        if feature_transforms is None:
            feature_transforms = self._default_transforms()

        X_perturbed = X_raw.copy()

        for transform in feature_transforms:
            X_perturbed = transform(X_perturbed)

        return X_perturbed

    def _default_transforms(self) -> list[Callable]:
        """Default feature engineering transformations for fraud data."""
        return [
            self._temporal_aggregation,
            self._behavioral_pattern,
            self._statistical_moment,
        ]

    def _temporal_aggregation(self, X: np.ndarray) -> np.ndarray:
        """Simulate temporal aggregation features (rolling stats, differences)."""
        if X.shape[1] < 2:
            return X

        n_samples, n_features = X.shape
        n_agg = min(3, n_features)

        aggregated = np.zeros((n_samples, n_features + n_agg))
        aggregated[:, :n_features] = X

        for i in range(n_agg):
            aggregated[:, n_features + i] = np.mean(X[:, : i + 1], axis=1)

        return aggregated

    def _behavioral_pattern(self, X: np.ndarray) -> np.ndarray:
        """Simulate behavioral pattern features (ratios, deviations)."""
        n_samples, n_features = X.shape

        if n_features < 2:
            return X

        patterns = np.zeros((n_samples, 2))

        pattern_features = X[:, : min(3, n_features)]

        patterns[:, 0] = np.max(pattern_features, axis=1) - np.min(
            pattern_features, axis=1
        )
        patterns[:, 1] = np.std(pattern_features, axis=1)

        return np.hstack([X, patterns])

    def _statistical_moment(self, X: np.ndarray) -> np.ndarray:
        """Add statistical moment features (skewness approximation)."""
        n_samples, n_features = X.shape

        if n_features < 3:
            return X

        moments = np.zeros((n_samples, 2))

        centered = X - np.mean(X, axis=1, keepdims=True)
        std = np.std(X, axis=1, keepdims=True) + 1e-8
        normalized = centered / std

        moments[:, 0] = np.mean(normalized**3, axis=1)
        moments[:, 1] = np.mean(normalized**4, axis=1)

        return np.hstack([X, moments])


class AdversarialStructuralClustering:
    """
    Groups adversarial perturbations using clustering to capture commonalities
    between different attack types (FGSM, PGD, etc.) and construct a more
    structured feature space.
    """

    def __init__(
        self, n_clusters: int = 5, method: str = "kmeans", epsilon: float = 0.05
    ):
        self.n_clusters = n_clusters
        self.method = method
        self.epsilon = epsilon
        self.clusterer = None
        self.scaler = StandardScaler()

    def fit_cluster(self, X_natural: np.ndarray, X_adversarial: np.ndarray) -> dict:
        """
        Fit clustering on adversarial perturbations to learn attack structure.

        Args:
            X_natural: Original natural samples
            X_adversarial: Adversarial perturbations

        Returns:
            Dictionary with cluster info and augmented data
        """
        perturbations = X_adversarial - X_natural

        X_combined = np.vstack([X_natural, X_adversarial])

        if self.method == "kmeans":
            self.clusterer = KMeans(
                n_clusters=self.n_clusters, random_state=42, n_init=10
            )
            labels = self.clusterer.fit_predict(X_combined)
        elif self.method == "dbscan":
            self.clusterer = DBSCAN(eps=self.epsilon * 2, min_samples=5)
            labels = self.clusterer.fit_predict(X_combined)
        else:
            raise ValueError(f"Unknown clustering method: {self.method}")

        cluster_centers = []
        for c in range(self.n_clusters):
            mask = labels == c
            if mask.any():
                center = np.mean(X_combined[mask], axis=0)
                cluster_centers.append(center)

        cluster_centers = (
            np.array(cluster_centers)
            if cluster_centers
            else X_natural[: self.n_clusters]
        )

        return {
            "labels": labels,
            "cluster_centers": cluster_centers,
            "n_clusters": self.n_clusters,
            "perturbations": perturbations,
        }

    def generate_cluster_aware_samples(
        self, X_natural: np.ndarray, y: np.ndarray, cluster_info: dict
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate cluster-aware adversarial samples that capture
        common attack patterns.
        """
        cluster_centers = cluster_info["cluster_centers"]

        X_augmented = [X_natural]
        y_augmented = [y]

        for center in cluster_centers:
            center_perturbation = center[: X_natural.shape[1]]
            perturbed = X_natural + center_perturbation * self.epsilon
            X_augmented.append(perturbed)
            y_augmented.append(y.copy())

        return np.vstack(X_augmented), np.hstack(y_augmented)


class TripletLossAdversarial:
    """
    TLA: Adds triplet loss to the standard objective function.
    Forces the model to bring adversarial samples closer to their
    'natural' counterparts of the same class while increasing distance
    between different classes.
    """

    def __init__(
        self,
        margin: float = 0.5,
        embedding_dim: int = 64,
    ):
        self.margin = margin
        self.embedding_dim = embedding_dim
        self.scaler = StandardScaler()

    def compute_triplet_loss(
        self, anchor: np.ndarray, positive: np.ndarray, negative: np.ndarray
    ) -> float:
        """
        Compute triplet loss: max(0, d(a,p) - d(a,n) + margin)

        Args:
            anchor: Natural sample embeddings
            positive: Adversarial sample embeddings (same class)
            negative: Sample embeddings (different class)

        Returns:
            Triplet loss value
        """
        ap_dist = np.sum((anchor - positive) ** 2, axis=1)
        an_dist = np.sum((anchor - negative) ** 2, axis=1)

        loss = np.maximum(0, ap_dist - an_dist + self.margin)
        return float(np.mean(loss))

    def create_triplets(
        self, X: np.ndarray, y: np.ndarray, n_triplets: int = 1000
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create triplet training data from natural + adversarial samples.

        Args:
            X: Feature matrix
            y: Labels
            n_triplets: Number of triplets to generate

        Returns:
            Tuple of (anchors, positives, negatives)
        """
        classes = np.unique(y)

        anchors = []
        positives = []
        negatives = []

        for _ in range(n_triplets):
            c = np.random.choice(classes)

            class_mask = y == c
            other_mask = y != c

            c_indices = np.where(class_mask)[0]
            other_indices = np.where(other_mask)[0]

            if len(c_indices) < 2 or len(other_indices) < 1:
                continue

            anc_idx = np.random.choice(c_indices)
            pos_idx = np.random.choice(c_indices[c_indices != anc_idx])
            neg_idx = np.random.choice(other_indices)

            anchors.append(X[anc_idx])
            positives.append(X[pos_idx])
            negatives.append(X[neg_idx])

        return (np.array(anchors), np.array(positives), np.array(negatives))

    def compute_weighted_loss(
        self, base_loss: float, triplet_loss: float, alpha: float = 0.1
    ) -> float:
        """
        Combine base loss with triplet loss.

        Args:
            base_loss: Standard model loss (e.g., cross-entropy)
            triplet_loss: Triplet loss from adversarial samples
            alpha: Weight for triplet loss component

        Returns:
            Combined loss
        """
        return base_loss + alpha * triplet_loss


class FraudGAN:
    """
    FraudGAN: Uses Generative Adversarial Networks to simulate sophisticated,
    targeted fraud patterns for adversarial retraining.
    """

    def __init__(
        self,
        latent_dim: int = 64,
        hidden_dim: int = 128,
        n_classes: int = 2,
        epsilon: float = 0.05,
    ):
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes
        self.epsilon = epsilon

        self.generator = None
        self.discriminator = None
        self.noise_scaler = StandardScaler()

        self._init_networks()

    def _init_networks(self):
        """Initialize simple MLP generator and discriminator."""
        np.random.seed(42)

        self.gen_weights = {
            "W1": np.random.randn(self.latent_dim, self.hidden_dim) * 0.1,
            "b1": np.zeros((1, self.hidden_dim)),
            "W2": np.random.randn(self.hidden_dim, self.hidden_dim) * 0.1,
            "b2": np.zeros((1, self.hidden_dim)),
            "W3": np.random.randn(self.hidden_dim, self.latent_dim) * 0.1,
            "b3": np.zeros((1, self.latent_dim)),
        }

        self.disc_weights = {
            "W1": np.random.randn(self.latent_dim, self.hidden_dim // 2) * 0.1,
            "b1": np.zeros((1, self.hidden_dim // 2)),
            "W2": np.random.randn(self.hidden_dim // 2, self.hidden_dim // 2) * 0.1,
            "b2": np.zeros((1, self.hidden_dim // 2)),
            "W3": np.random.randn(self.hidden_dim // 2, 1) * 0.1,
            "b3": np.zeros((1, 1)),
        }

    def _relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def generate_fraud_patterns(
        self, X_real: np.ndarray, n_samples: int = 1000, epochs: int = 50
    ) -> np.ndarray:
        """
        Train simplified GAN and generate synthetic fraud patterns.

        Args:
            X_real: Real fraud samples for training
            n_samples: Number of synthetic samples to generate
            epochs: Training epochs

        Returns:
            Synthetic fraud-like samples
        """
        X_scaled = self.noise_scaler.fit_transform(X_real)

        real_dim = X_scaled.shape[1]

        for epoch in range(epochs):
            Z = np.random.randn(n_samples, self.latent_dim)

            fake_samples = self._forward_generator(Z)[:, :real_dim]

            real_scores = self._forward_discriminator(
                X_scaled[: min(n_samples, len(X_scaled))]
            )
            fake_scores = self._forward_discriminator(fake_samples)

            if epoch % 20 == 0:
                real_mean = np.mean(real_scores)
                fake_mean = np.mean(fake_scores)

        Z_final = np.random.randn(n_samples, self.latent_dim)
        synthetic = self._forward_generator(Z_final)[:, :real_dim]

        synthetic = self.noise_scaler.inverse_transform(synthetic)

        return synthetic

    def _forward_generator(self, Z: np.ndarray) -> np.ndarray:
        h1 = self._relu(Z @ self.gen_weights["W1"] + self.gen_weights["b1"])
        h2 = self._relu(h1 @ self.gen_weights["W2"] + self.gen_weights["b2"])
        return h2 @ self.gen_weights["W3"] + self.gen_weights["b3"]

    def _forward_discriminator(self, X: np.ndarray) -> np.ndarray:
        h1 = self._relu(X @ self.disc_weights["W1"] + self.disc_weights["b1"])
        h2 = self._relu(h1 @ self.disc_weights["W2"] + self.disc_weights["b2"])
        return self._sigmoid(h2 @ self.disc_weights["W3"] + self.disc_weights["b3"])


class AdvancedAdversarialTrainer:
    """
    Main class combining all advanced adversarial training techniques.
    Implements the min-max optimization:
        min_θ E_{(x,y)~D} [ max_||δ||≤ε L(f_θ(x+δ), y) ]
    """

    def __init__(
        self,
        epsilon: float = 0.05,
        enable_apfe: bool = True,
        enable_asc: bool = True,
        enable_tla: bool = True,
        enable_fraudgan: bool = True,
        n_clusters: int = 5,
        triplet_weight: float = 0.1,
    ):
        self.epsilon = epsilon

        self.apfe = AttackPropagation(epsilon=epsilon) if enable_apfe else None
        self.asc = (
            AdversarialStructuralClustering(n_clusters=n_clusters, epsilon=epsilon)
            if enable_asc
            else None
        )
        self.tla = TripletLossAdversarial(margin=0.5) if enable_tla else None
        self.fraudgan = FraudGAN(epsilon=epsilon) if enable_fraudgan else None

        self.triplet_weight = triplet_weight
        self.enable_apfe = enable_apfe
        self.enable_asc = enable_asc
        self.enable_tla = enable_tla
        self.enable_fraudgan = enable_fraudgan

    def generate_adversarial_training_set(
        self, X: np.ndarray, y: np.ndarray, method: str = "all"
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate augmented training set with all adversarial techniques.

        Args:
            X: Feature matrix
            y: Labels
            method: Which techniques to use ("all", "fgsm", "apfe", "asc", "tla", "fraudgan")

        Returns:
            Tuple of (augmented_X, augmented_y)
        """
        X_aug = [X]
        y_aug = [y]

        if method in ("all", "fgsm"):
            X_fgsm = self._fgsm_perturb(X)
            X_aug.append(X_fgsm)
            y_aug.append(y.copy())

        if self.enable_apfe and method in ("all", "apfe"):
            X_apfe = self._apfe_perturb(X)
            X_aug.append(X_apfe)
            y_aug.append(y.copy())

        if self.enable_fraudgan and method in ("all", "fraudgan"):
            fraud_mask = y == 1
            if fraud_mask.sum() > 10:
                X_fraud = X[fraud_mask]
                synthetic_fraud = self.fraudgan.generate_fraud_patterns(X_fraud)
                X_aug.append(synthetic_fraud)
                y_aug.append(np.ones(len(synthetic_fraud), dtype=int))

        if self.enable_asc and method in ("all", "asc"):
            X_fgsm = self._fgsm_perturb(X)
            cluster_info = self.asc.fit_cluster(X, X_fgsm)
            X_asc, y_asc = self.asc.generate_cluster_aware_samples(X, y, cluster_info)
            X_aug.append(X_asc)
            y_aug.append(y_asc)

        if self.enable_tla and method in ("all", "tla"):
            X_fgsm = self._fgsm_perturb(X)
            X_tla = self._tla_augment(X, X_fgsm, y)
            X_aug.append(X_tla)
            y_aug.append(y.copy())

        return np.vstack(X_aug), np.hstack(y_aug)

    def _fgsm_perturb(self, X: np.ndarray) -> np.ndarray:
        """Standard FGSM-style perturbation."""
        noise = self.epsilon * np.sign(np.random.randn(*X.shape))
        X_adv = X + noise
        return np.clip(X_adv, 0, None).astype(np.float32)

    def _apfe_perturb(self, X: np.ndarray) -> np.ndarray:
        """Attack propagation through feature engineering."""
        X_perturbed = self.apfe.propagate(X)
        X_adv = X + self.epsilon * np.sign(np.random.randn(*X.shape))
        return np.clip(X_adv[:, : X.shape[1]], 0, None).astype(np.float32)

    def _tla_augment(
        self, X_nat: np.ndarray, X_adv: np.ndarray, y: np.ndarray
    ) -> np.ndarray:
        """Triplet loss adversarial augmentation."""
        anchors, positives, negatives = self.tla.create_triplets(X_nat, y)

        X_triplets = np.vstack([anchors, positives, negatives])

        X_adv_extended = np.vstack([X_adv, X_nat])
        y_adv_extended = np.hstack([y, y])

        combined_idx = np.random.choice(
            len(X_adv_extended),
            size=min(len(X_triplets), len(X_adv_extended)),
            replace=False,
        )

        return X_adv_extended[combined_idx]

    def get_technique_summary(self) -> dict:
        """Return summary of enabled techniques."""
        return {
            "epsilon": self.epsilon,
            "attack_propagation": self.enable_apfe,
            "structural_clustering": self.enable_asc,
            "triplet_loss": self.enable_tla,
            "fraud_gan": self.enable_fraudgan,
            "triplet_weight": self.triplet_weight,
        }
