import ntpath
import numpy as np
import os

from elliot.recommender import ProxyRecommender
from elliot.recommender.base_recommender_model import init_charger

import wandb

from .calibrated_reranking import CalibratedReranking


class CalibratedReranker(ProxyRecommender):
    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):
        """Greedily rerank saved recommendations, trading off relevance and emotion calibration."""
        self._random = np.random

        self._fold_index = None

        self._params_list = [
            ("_input_path", "input_path", "input_path", "", None, None),
            ("_lambda", "lambda", "lam", 0.5, None, None),
            ("_k", "k", "k", 10, None, None),
            ("_modalitiy", "modalitiy", "modality", "emotion", None, None),
            ("_loader", "loader", "load", "EmotionAttribute", None, None),
            ("_calibration_metric", "calibration_metric", "calib", "mahalanobis", None, None),
        ]
        self.autoset_params()

        self._ratings = self._data.train_dict

        self.__setattr__(f'''_side_{self._modalitiy}''',
                         self._data.side_information.__getattribute__(f'''{self._loader}'''))

        multimodal_feature = self.__getattribute__(f'''_side_{self._modalitiy}''').object.get_all_features(item_order=self._data.items)

        self._reranking_model = CalibratedReranking(data=self._data,
                                 k=self._k,
                                 modalitiy=self._modalitiy,
                                 multimodal_feature=multimodal_feature,
                                 lam=self._lambda,
                                 calibration_metric=self._calibration_metric)

        self._base_model_name = ntpath.basename(self._input_path).rsplit(".",1)[0].split('_')[0]

        self.log_wandb = getattr(self._params, 'log_wandb', False)
        if self.log_wandb:
            wandb.init(
                project=f"CalibratedReranker{self._base_model_name}-{os.path.basename(os.path.dirname(config.data_config.dataset_path))}",
                name=self.name,
                config={
                    **{
                        "lambda": self._lambda,
                        "k": self._k,
                        "modalitiy": self._modalitiy,
                        "calibration_metric": self._calibration_metric,
                        "path": ntpath.basename(self._input_path)
                    }
                },
                reinit=True,
            )

    @property
    def name(self):
        return f"CalibratedReranker{self._base_model_name}_{self._lambda}_{self._k}_{self._modalitiy}_{self._calibration_metric}"

    def train(self):
        print("Reading recommendations")

        if self._fold_index is not None:
            fold_input_path = self._input_path.replace("fold=0", f"fold={self._fold_index}")
            recs = self.read_recommendations(fold_input_path)
        else:
            recs = self.read_recommendations(self._input_path)

        self._recommendations = self._reranking_model.rerank_recommendations(recs)

        print("Evaluating recommendations")
        self.evaluate(wandb_log=self.log_wandb)
