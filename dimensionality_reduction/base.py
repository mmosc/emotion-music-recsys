"""Abstract base class for modality feature dimensionality reduction."""

from abc import ABC, abstractmethod

import numpy as np


class DimensionalityReducer(ABC):
    """Reduces the dimensionality of a modality feature matrix."""

    def __init__(self, n_components: int):
        self.n_components = n_components

    @property
    @abstractmethod
    def name(self) -> str:
        """Short tag used to name output folders/files (e.g. 'pca')."""

    @abstractmethod
    def fit(self, features: np.ndarray) -> "DimensionalityReducer":
        """Fit the reducer on a (n_items, original_dim) feature matrix."""

    @abstractmethod
    def transform(self, features: np.ndarray) -> np.ndarray:
        """Project a (n_items, original_dim) matrix to (n_items, n_components)."""

    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        return self.fit(features).transform(features)
