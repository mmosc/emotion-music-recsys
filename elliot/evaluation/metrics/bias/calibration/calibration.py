"""
This is the implementation of the Mahalanobis Calibration metric.
It proceeds from a user-wise computation, and averages the values over the users.
"""

__version__ = '0.3.1'

import numpy as np
from scipy.spatial.distance import mahalanobis

from elliot.evaluation.metrics.base_metric import BaseMetric


class Calibration(BaseMetric):
    r"""
    Mahalanobis Calibration

    Measures how far a user's top-K recommendation list drifts from that user's own
    taste profile in some continuous feature space (e.g. emotion embeddings), using the
    same Mahalanobis-distance formulation as the greedy calibrated reranker
    (Steck, "Calibrated Recommendations", RecSys 2018, adapted to a continuous profile
    instead of a categorical genre distribution): the distance between the mean feature
    vector of the recommended list and the mean feature vector of the user's training
    history, under the inverse covariance of the (training-set) item feature space.

    Lower values mean the recommendations are better calibrated to the user's history;
    higher values mean the list has drifted further from what the user has historically
    consumed.

    To compute the metric, add it to the config file adopting the following pattern:

    .. code:: yaml

        complex_metrics:
        - metric: Calibration
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
        # training-set only, same reference population the calibrated reranker uses
        # for both the user's profile and the covariance of the feature space
        self._train_dict = self._data.train_dict
        self._train_items = self._data.items

        feature_folder_path = additional_data["feature_data"]
        self._feature_map = Calibration._load_features(feature_folder_path, self._train_items)

        feature_matrix = np.array([self._feature_map[i] for i in self._train_items])
        covariance = np.cov(feature_matrix.T)
        self._inv_covariance = np.linalg.pinv(covariance)

    @staticmethod
    def name():
        """
        Metric Name Getter
        :return: returns the public name of the metric
        """
        return "Calibration"

    @staticmethod
    def _load_features(feature_folder_path, items):
        return {i: np.load(f"{feature_folder_path}/{i}.npy") for i in items}

    def __user_calibration(self, user, user_recommendations, cutoff):
        """
        Per User Mahalanobis Calibration
        :param user: user id
        :param user_recommendations: list of user recommendation in the form [(item1,value1),...]
        :param cutoff: numerical threshold to limit the recommendation list
        :return: the Mahalanobis distance between the recommended list's and the user's mean
                 training-history feature profile
        """
        rec_items = [i for i, _ in user_recommendations[:cutoff] if i in self._feature_map]
        history_items = [i for i in self._train_dict.get(user, {}).keys() if i in self._feature_map]

        if not rec_items or not history_items:
            return 0.0

        rec_profile = np.mean([self._feature_map[i] for i in rec_items], axis=0)
        user_profile = np.mean([self._feature_map[i] for i in history_items], axis=0)

        diff = rec_profile - user_profile
        if np.allclose(diff, 0):
            return 0.0

        return mahalanobis(rec_profile, user_profile, self._inv_covariance)

    def eval_user_metric(self):
        """
        Evaluation function
        :return: the overall Mahalanobis Calibration value per user
        """
        return {u: self.__user_calibration(u, u_r, self._cutoff)
                for u, u_r in self._recommendations.items()}
