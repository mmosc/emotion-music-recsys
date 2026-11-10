import ntpath
import os

import numpy as np

from elliot.recommender import ProxyRecommender
from elliot.recommender.base_recommender_model import init_charger

import wandb

from .random_shuffle import RandomShuffle


class RandomShuffleReranker(ProxyRecommender):
    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):
        """Rerank a base model's saved top-K list uniformly at random."""
        self._random = np.random

        self._fold_index = None

        self._params_list = [
            ("_input_path", "input_path", "input_path", "", None, None),
            ("_k", "k", "k", 10, None, None),
            ("_seed", "seed", "seed", 42, None, None),
        ]
        self.autoset_params()

        self._ratings = self._data.train_dict

        self._reranking_model = RandomShuffle(data=self._data, k=self._k, seed=self._seed)

        self._base_model_name = ntpath.basename(self._input_path).rsplit(".", 1)[0].split('_')[0]

        self.log_wandb = getattr(self._params, 'log_wandb', False)
        if self.log_wandb:
            wandb.init(
                project=f"RandomShuffle{self._base_model_name}-"
                        f"{os.path.basename(os.path.dirname(config.data_config.dataset_path))}",
                name=self.name,
                config={
                    "k": self._k,
                    "seed": self._seed,
                    "path": ntpath.basename(self._input_path),
                },
                reinit=True,
            )

    @property
    def name(self):
        return f"RandomShuffle{self._base_model_name}_{self._k}_{self._seed}"

    def train(self):
        print("Reading recommendations")

        if self._fold_index is not None:
            fold_input_path = self._input_path.replace("fold=0", f"fold={self._fold_index}")
            recs = self.read_recommendations(fold_input_path)
        else:
            recs = self.read_recommendations(self._input_path)

        self._recommendations = self._reranking_model.rerank_recommendations(
            recs, fold=self._fold_index or 0)

        print("Evaluating recommendations")
        self.evaluate(wandb_log=self.log_wandb)
