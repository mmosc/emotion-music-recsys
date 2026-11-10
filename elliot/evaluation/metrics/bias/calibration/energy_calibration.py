"""
This is the implementation of the Energy-Distance Calibration metric.
It proceeds from a user-wise computation, and averages the values over the users.
"""

__version__ = '0.3.1'

import numpy as np
from scipy.spatial.distance import cdist

from elliot.evaluation.metrics.base_metric import BaseMetric


class EnergyCalibration(BaseMetric):
    r"""
    Energy-Distance Calibration

    Measures how far a user's top-K recommendation list's feature distribution
    (e.g. GEMS emotion embeddings) drifts from that user's own training-history
    distribution, using the (squared) energy distance between the two samples
    (Szekely & Rizzo):

        E(T,R) = 2 * mean_{t in T, r in R} d(t,r)
                   - mean_{t,t' in T} d(t,t')
                   - mean_{r,r' in R} d(r,r')

    Unlike the Mahalanobis :class:`Calibration` metric, which reduces the
    recommendation list and the training history to their mean feature vector
    before comparing, this compares the two full empirical distributions --
    so a list that is unusually clustered or spread out relative to the
    user's history is penalized even when its mean matches. Features are
    standardized per-dimension using the training-history statistics before
    computing distances, so no single coordinate dominates.

    Lower values mean the recommendations are better calibrated (their
    feature distribution is closer to the user's history); higher values
    mean the list has drifted further from what the user historically
    consumed. Finite-sample energy-distance estimates can dip slightly below
    zero even for identical distributions; these are clipped to 0.

    To compute the metric, add it to the config file adopting the following pattern:

    .. code:: yaml

        complex_metrics:
        - metric: EnergyCalibration
          feature_data: /path/to/emotion_embeddings/gems
    """

    def __init__(self, recommendations, config, params, eval_objects, additional_data):
        """
        Constructor
        :param recommendations: list of recommendations in the form {user: [(item1,value1),...]}
        :param config: SimpleNameSpace that represents the configuration of the experiment
        :param params: Parameters of the model
        :param eval_objects: list of objects that may be useful for the computation of the different metrics
        :param additional_data: dictionary with the additional data required by the metric (feature_data path)
        """
        super().__init__(recommendations, config, params, eval_objects, additional_data)
        self._cutoff = self._evaluation_objects.cutoff

        self._data = self._evaluation_objects.data
        # training-set only, same reference population the Mahalanobis Calibration
        # metric and the calibrated reranker use for the user's profile
        self._train_dict = self._data.train_dict
        self._train_items = self._data.items

        feature_folder_path = additional_data["feature_data"]
        self._feature_map = EnergyCalibration._load_features(feature_folder_path, self._train_items)

        feature_matrix = np.array([self._feature_map[i] for i in self._train_items])
        self._mu = feature_matrix.mean(axis=0)
        sd = feature_matrix.std(axis=0)
        self._sd = np.where(sd < 1e-12, 1.0, sd)  # guard against constant dimensions

    @staticmethod
    def name():
        """
        Metric Name Getter
        :return: returns the public name of the metric
        """
        return "EnergyCalibration"

    @staticmethod
    def _load_features(feature_folder_path, items):
        return {i: np.load(f"{feature_folder_path}/{i}.npy") for i in items}

    def __user_energy_calibration(self, user, user_recommendations, cutoff):
        """
        Per User Energy-Distance Calibration
        :param user: user id
        :param user_recommendations: list of user recommendation in the form [(item1,value1),...]
        :param cutoff: numerical threshold to limit the recommendation list
        :return: the energy distance between the recommended list's and the user's
                 training-history feature distributions
        """
        rec_items = [i for i, _ in user_recommendations[:cutoff] if i in self._feature_map]
        history_items = [i for i in self._train_dict.get(user, {}).keys() if i in self._feature_map]

        if not rec_items or not history_items:
            return 0.0

        T = (np.array([self._feature_map[i] for i in history_items]) - self._mu) / self._sd
        R = (np.array([self._feature_map[i] for i in rec_items]) - self._mu) / self._sd

        d_tr = cdist(T, R).mean()
        d_tt = cdist(T, T).mean()
        d_rr = cdist(R, R).mean()

        energy = 2.0 * d_tr - d_tt - d_rr
        return max(energy, 0.0)

    def eval_user_metric(self):
        """
        Evaluation function
        :return: the overall Energy-Distance Calibration value per user
        """
        return {u: self.__user_energy_calibration(u, u_r, self._cutoff)
                for u, u_r in self._recommendations.items()}
