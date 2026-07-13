"""Hand gesture trainer that builds a KNN classifier for custom gestures."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sklearn.neighbors import KNeighborsClassifier


class GestureTrainer:
    """Builds and saves a gesture recognition model."""

    def __init__(self, n_neighbors: int = 3) -> None:
        self.n_neighbors = n_neighbors

    def train(
        self,
        gesture_data_dir: str | Path = "gesture_data",
        output_path: str | Path = "models/gesture_model.pkl",
        verbose: bool = True,
    ) -> None:
        """Train a KNN classifier on collected gesture samples."""
        gesture_data_dir = Path(gesture_data_dir)
        output_path = Path(output_path)

        if not gesture_data_dir.exists():
            raise FileNotFoundError(
                f"Gesture data directory not found: {gesture_data_dir}\n"
                "Run gesture_onboard.py to capture training data first."
            )

        samples = []
        labels = []
        class_names = []

        for gesture_dir in sorted(gesture_data_dir.iterdir()):
            if not gesture_dir.is_dir():
                continue

            feature_files = sorted(gesture_dir.glob("*.npy"))
            if not feature_files:
                if verbose:
                    print(f"  [skip] {gesture_dir.name}: no samples found.")
                continue

            class_index = len(class_names)
            class_names.append(gesture_dir.name)

            if verbose:
                print(f"Loading {gesture_dir.name} ({len(feature_files)} sample(s))...")

            for npy_file in feature_files:
                features = np.load(npy_file).astype(np.float32)
                samples.append(features)
                labels.append(class_index)

            if verbose:
                print(f"  -> {len(feature_files)} sample(s) loaded.")

        if len(samples) < self.n_neighbors:
            raise ValueError(
                f"Need at least {self.n_neighbors} training samples, got {len(samples)}. "
                "Capture more gesture samples."
            )

        X = np.array(samples, dtype=np.float32)
        y = np.array(labels, dtype=np.int32)

        model = KNeighborsClassifier(n_neighbors=self.n_neighbors)
        model.fit(X, y)

        distance_thresholds = self._build_distance_thresholds(model, X, y, class_names)
        global_threshold = max(distance_thresholds.values()) if distance_thresholds else 0.35

        one_class_profile = None
        if len(class_names) == 1 and class_names[0].lower() == "palm":
            one_class_profile = self._build_one_class_profile(X)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as file_obj:
            pickle.dump(
                {
                    "model": model,
                    "classes": class_names,
                    "distance_thresholds": distance_thresholds,
                    "global_threshold": global_threshold,
                    "one_class_profile": one_class_profile,
                },
                file_obj,
            )

        if verbose:
            print(f"\nModel trained on {len(X)} samples across {len(class_names)} gesture(s)")
            print(f"Distance thresholds: {distance_thresholds}")
            if one_class_profile is not None:
                print(f"One-class palm threshold: {one_class_profile['max_distance']:.4f}")
            print(f"Saved to {output_path}")

    def _build_distance_thresholds(self, model, X, y, class_names):
        neighbor_count = min(len(X), max(2, self.n_neighbors + 1))
        distances, indices = model.kneighbors(X, n_neighbors=neighbor_count)
        class_scores = {name: [] for name in class_names}

        for row_index, (row_distances, row_indices) in enumerate(zip(distances, indices)):
            same_class_distances = []
            for distance, neighbor_index in zip(row_distances, row_indices):
                if neighbor_index == row_index:
                    continue
                if y[neighbor_index] == y[row_index]:
                    same_class_distances.append(float(distance))
                if len(same_class_distances) >= self.n_neighbors:
                    break

            if not same_class_distances:
                continue

            class_name = class_names[y[row_index]]
            class_scores[class_name].append(float(np.mean(same_class_distances)))

        thresholds = {}
        for class_name, scores in class_scores.items():
            if not scores:
                thresholds[class_name] = 0.35
                continue

            values = np.array(scores, dtype=np.float32)
            threshold = float(np.percentile(values, 95) + 0.03)
            thresholds[class_name] = max(0.12, min(threshold, 0.6))

        return thresholds

    def _build_one_class_profile(self, X: np.ndarray) -> dict:
        """Build centroid-distance profile for palm-only training.

        This provides an explicit rejection boundary for non-palm objects.
        """
        centroid = np.mean(X, axis=0).astype(np.float32)
        distances = np.linalg.norm(X - centroid, axis=1)
        # 97th percentile + small margin keeps palm variance while rejecting outliers.
        max_distance = float(np.percentile(distances, 97) + 0.03)
        return {
            "centroid": centroid,
            "max_distance": max_distance,
        }
