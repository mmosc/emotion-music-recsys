"""PCA-based dimensionality reduction for modality features."""

import numpy as np
from sklearn.decomposition import PCA

from dimensionality_reduction.base import DimensionalityReducer


class PCAReducer(DimensionalityReducer):
    def __init__(self, n_components: int, random_state: int = 42):
        super().__init__(n_components)
        self._pca = PCA(n_components=n_components, random_state=random_state)

    @property
    def name(self) -> str:
        return "pca"

    def fit(self, features: np.ndarray) -> "PCAReducer":
        self._pca.fit(features)
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        return self._pca.transform(features)
