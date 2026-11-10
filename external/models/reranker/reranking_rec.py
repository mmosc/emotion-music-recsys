import ntpath
import numpy as np
import pandas as pd
from tqdm import tqdm
import os

from elliot.recommender import ProxyRecommender
from elliot.recommender.base_recommender_model import init_charger

import wandb

from .reranking_similarity import Similarity


class RerankingRecommender(ProxyRecommender):
    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):
        """Create a reranking recommender based on emotion to rerank already generated
        recommendations.
        """
        self._random = np.random

        self._fold_index = None

        self._params_list = [
            ("_input_path", "input_path", "input_path", "", None, None),
            ("_similarity", "similarity", "sim", "cosine", None, None),
            ("_k", "k", "k", 10, None, None),
            ("_modalitiy", "modalitiy", "modality", "emotion", None, None),
            ("_loader", "loader", "load", "EmotionAttribute", None, None)
        ]
        self.autoset_params()

        self._ratings = self._data.train_dict

        self.__setattr__(f'''_side_{self._modalitiy}''',
                         self._data.side_information.__getattribute__(f'''{self._loader}'''))

        multimodal_feature = self.__getattribute__(f'''_side_{self._modalitiy}''').object.get_all_features(item_order=self._data.items)

        self._reranking_model = Similarity(data=self._data,
                                 similarity=self._similarity,
                                 k=self._k,
                                 modalitiy=self._modalitiy,
                                 multimodal_feature=multimodal_feature)

        self._base_model_name = ntpath.basename(self._input_path).rsplit(".",1)[0].split('_')[0]

        self.log_wandb = getattr(self._params, 'log_wandb', False)
        if self.log_wandb:
            wandb.init(
                project=f"Reranker{self._base_model_name}-{os.path.basename(os.path.dirname(config.data_config.dataset_path))}",
                name=self.name,
                config={
                    **{
                        "similarity": self._similarity,
                        "k": self._k,
                        "modalitiy": self._modalitiy,
                        "path": ntpath.basename(self._input_path)
                    }
                },
                reinit=True,
            )

    @property
    def name(self):
        return f"Reranker{self._base_model_name}_{self._similarity}_{self._k}_{self._modalitiy}"

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


class RerankingRecommender2(RerankingRecommender):
    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):
        """Create a reranking recommender based on emotion to rerank already generated
        recommendations.
        """
        super().__init__(data, config, params, *args, **kwargs)