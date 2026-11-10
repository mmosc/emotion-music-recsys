from tqdm import tqdm
import numpy as np
import torch
import os

from elliot.utils.write import store_recommendation
from .custom_sampler import Sampler
from elliot.recommender import BaseRecommenderModel
from elliot.recommender.base_recommender_model import init_charger
from elliot.recommender.recommender_utils_mixin import RecMixin
from .LightGCNModel import LightGCNModel

from torch_sparse import SparseTensor

import math
import wandb


class LightGCN(RecMixin, BaseRecommenderModel):
    r"""LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation"""
    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):
        ######################################

        self._fold_index = None

        self._params_list = [
            ("_learning_rate", "lr", "lr", 0.0005, float, None),
            ("_factors", "factors", "factors", 64, int, None),
            ("_l_w", "l_w", "l_w", 0.01, float, None),
            ("_n_layers", "n_layers", "n_layers", 1, int, None),
            ("_normalize", "normalize", "normalize", True, bool, None)
        ]
        self.autoset_params()

        self._sampler = Sampler(self._data.i_train_dict, seed=self._seed)
        if self._batch_size < 1:
            self._batch_size = self._num_users

        row, col = data.sp_i_train.nonzero()
        col = [c + self._num_users for c in col]
        edge_index = np.array([row, col])
        edge_index = torch.tensor(edge_index, dtype=torch.int64)
        self.adj = SparseTensor(row=torch.cat([edge_index[0], edge_index[1]], dim=0),
                                col=torch.cat([edge_index[1], edge_index[0]], dim=0),
                                sparse_sizes=(self._num_users + self._num_items,
                                              self._num_users + self._num_items))

        self._model = LightGCNModel(
            num_users=self._num_users,
            num_items=self._num_items,
            learning_rate=self._learning_rate,
            embed_k=self._factors,
            l_w=self._l_w,
            n_layers=self._n_layers,
            adj=self.adj,
            normalize=self._normalize,
            random_seed=self._seed
        )

        self.log_wandb = getattr(self._params, 'log_wandb', False)
        if self.log_wandb:
            wandb.init(
                project=f"LightGCN-{config.data_config.dataset_path.split('/')[-2]}",
                name=self.name,
                config={
                    **{
                        "learning_rate": self._learning_rate,
                        "batch_size": self._batch_size,
                        "factors": self._factors,
                        "l_w": self._l_w,
                        "n_layers": self._n_layers,
                        "normalize": self._normalize,
                        "seed": self._seed
                    }
                },
                reinit=True,
            )

    @property
    def name(self):
        return "LightGCN" \
               + f"_{self.get_base_params_shortcut()}" \
               + f"_{self.get_params_shortcut()}"

    def train(self):
        if self._restore:
            return self.restore_weights()

        start_epoch = self.load_resume_state() if self._warm_restart else 0
        for it in self.iterate(self._epochs, start=start_epoch):
            loss = 0
            steps = 0
            n_batch = int(self._data.transactions / self._batch_size) if self._data.transactions % self._batch_size == 0 else int(self._data.transactions / self._batch_size) + 1
            with tqdm(total=n_batch, disable=not self._verbose) as t:
                for batch in self._sampler.step(self._data.transactions, self._batch_size):
                    steps += 1
                    loss += self._model.train_step(batch)

                    if math.isnan(loss) or math.isinf(loss) or (not loss):
                        break

                    t.set_postfix({'loss': f'{loss / steps:.5f}'})
                    t.update()

            self.evaluate(it, loss / (it + 1))

        self._finalize_best()

        if self._warm_restart:
            self.clear_resume_state()

    def get_recommendations(self, k: int = 100):
        predictions_top_k_test = {}
        predictions_top_k_val = {}
        gu, gi = self._model.propagate_embeddings(evaluate=True)
        for index, offset in enumerate(range(0, self._num_users, self._batch_size)):
            offset_stop = min(offset + self._batch_size, self._num_users)
            predictions = self._model.predict(gu[offset: offset_stop], gi)
            recs_val, recs_test = self.process_protocol(k, predictions, offset, offset_stop)
            predictions_top_k_val.update(recs_val)
            predictions_top_k_test.update(recs_test)
        return predictions_top_k_val, predictions_top_k_test

    def get_single_recommendation(self, mask, k, predictions, offset, offset_stop):
        v, i = self._model.get_top_k(predictions, mask[offset: offset_stop], k=k)
        items_ratings_pair = [list(zip(map(self._data.private_items.get, u_list[0]), u_list[1]))
                              for u_list in list(zip(i.detach().cpu().numpy(), v.detach().cpu().numpy()))]
        return dict(zip(map(self._data.private_users.get, range(offset, offset_stop)), items_ratings_pair))

    def evaluate(self, it=None, loss=0):
        if (it is None) or (not (it + 1) % self._validation_rate):
            recs = self.get_recommendations(self.evaluator.get_needed_recommendations())
            defer_test = (self._defer_test_eval and self._save_weights
                          and it is not None)
            result_dict = self.evaluator.eval(
                recs, sides=("val",) if defer_test else ("val", "test"))

            self._losses.append(loss)

            prev_val_metric = (
                self._results[-2][self._validation_k]["val_results"][self._validation_metric]
                if len(self._results) > 1 else None
            )

            self._results.append(result_dict)

            if it is not None:
                val_metric = result_dict[self._validation_k]["val_results"][self._validation_metric]
                delta_str = f" ({val_metric - prev_val_metric:+.5f})" if prev_val_metric is not None else ""
                fold_prefix = f"fold={self._fold_index} " if getattr(self, "_fold_index", None) is not None else ""
                self.logger.info(
                    f'{fold_prefix}Epoch {(it + 1)}/{self._epochs} loss {loss/(it + 1):.5f} '
                    f'val_{self._validation_metric}@{self._validation_k} {val_metric:.5f}{delta_str}'
                )
                if self.log_wandb:
                    log_payload = {
                        "epochs": it + 1,
                        # for now it does not allow logging the two losses separately
                        "loss": loss / (it + 1),
                        # For now, we only log one metric
                        "val_ndcg5": result_dict[5]['val_results']['nDCG'],
                    }
                    if not defer_test:
                        log_payload["test_ndcg5"] = result_dict[5]['test_results']['nDCG']
                    wandb.log(log_payload)
            else:
                self.logger.info(f'Finished')

            if (len(self._results) - 1) == self.get_best_arg():
                if it is not None:
                    self._params.best_iteration = it + 1
                self.logger.info("******************************************")
                self.best_metric_value = self._results[-1][self._validation_k]["val_results"][self._validation_metric]
                if self._save_weights:
                    if hasattr(self, "_model"):
                        torch.save({
                            'model_state_dict': self._model.state_dict(),
                            'optimizer_state_dict': self._model.optimizer.state_dict()
                        }, self._saving_filepath)
                    else:
                        self.logger.warning("Saving weights FAILED. No model to save.")

                # meta.defer_recs: written once in _finalize_best() instead
                # of on every improvement epoch (needs the best-weights file).
                if self._save_recs and not (self._defer_recs and self._save_weights):
                    self.logger.info(f"Writing recommendations at: {self._config.path_output_rec_result}")
                    store_recommendation(recs[1], self._recs_tsv_path())

            if (self._warm_restart and it is not None
                    and not (it + 1) % self._resume_save_rate):
                self.save_resume_state(it)

    def restore_weights(self):
        try:
            checkpoint = torch.load(self._saving_filepath)
            self._model.load_state_dict(checkpoint['model_state_dict'])
            self._model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"Model correctly Restored")
            self.evaluate()
            return True

        except Exception as ex:
            raise Exception(f"Error in model restoring operation! {ex}")

        return False