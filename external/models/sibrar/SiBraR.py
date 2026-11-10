import hashlib
import itertools
import numpy as np
import torch
import os
from tqdm import tqdm
from ast import literal_eval as make_tuple
from queue import Queue
from threading import Thread
# ToDo check if we need a different custom sampler.
# Right now it is the same of CLCRec
from .custom_sampler import Sampler

from elliot.utils.write import store_recommendation

from elliot.recommender import BaseRecommenderModel
from .SiBraRModel import SiBraRModel
from elliot.recommender.recommender_utils_mixin import RecMixin
from elliot.recommender.base_recommender_model import init_charger
import math

#     os.environ["WANDB_MODE"] = "disabled"


def build_recs_filename(name, fold_index):
    """Recs TSV filename for a (model, fold); long names are md5-truncated."""
    fold_str = f"_fold={fold_index}" if fold_index is not None else ""
    full_name = f"{name}{fold_str}.tsv"
    if len(full_name) > 240:
        suffix = fold_str + ".tsv"
        h = hashlib.md5(full_name.encode()).hexdigest()[:8]
        full_name = full_name[:240 - len(suffix) - 9] + f"_{h}{suffix}"
    return full_name


def prefetch_to_pinned(iterable, n_prefetch=2):
    """Prefetch sampler batches as pinned CPU tensors on a background thread."""
    pin = torch.cuda.is_available()

    def convert(arr):
        t = torch.from_numpy(arr)
        return t.pin_memory() if pin else t

    q = Queue(maxsize=n_prefetch)
    sentinel = object()

    def producer():
        for user, pos, neg in iterable:
            q.put((convert(user), convert(pos), convert(neg)))
        q.put(sentinel)

    t = Thread(target=producer, daemon=True)
    t.start()
    while (item := q.get()) is not sentinel:
        yield item


class SiBraR(RecMixin, BaseRecommenderModel):
    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):
        # Define Parameters as tuples: (variable_name, public_name, shortcut, default, reading_function, printing_function)
        self._params_list = [
            ("_lr", "lr", "lr", 0.0005, float, None),
            ("_num_neg", "num_neg", "num_neg", 128, int, None),
            ("_input_dim", "input_dim", "input_dim", 256, int, None),
            ("_mid_layers", "mid_layers", "mid_layers", [], list, None),
            ("_emb_dim", "emb_dim", "emb_dim", 64, int, None),
            ("_b_norm", "b_norm", "b_norm", False, bool, None),
            ("_w_decay", "w_decay", "w_decay", 0.01, float, None),
            ("_cl_weight", "cl_weight", "cl_weight", 0.001, float, None),
            ("_u_prof", "u_prof", "u_prof", True, bool, None),
            ("_norm_input_feat", "norm_input_feat", "norm_input_feat", False, bool, None),
            ("_dropout", "dropout", "dropout", -1., float, None),
            ("_norm_sbra_input", "norm_sbra_input", "norm_sbra_input", False, bool, None),
            ("_cl_temp", "cl_temp", "cl_temp", 0.01, float, None),
            ("_cl_force_interactions_modality", "cl_force_interactions_modality", "cl_force_int_mod", False, bool, None),
            ("_cl_always_interactions_vs_agg", "cl_always_interactions_vs_agg", "cl_int_vs_agg", False, bool, None),
            ("_agg_method", "agg_method", "agg_method", 'mean', str, None),
            ("_modalities", "modalities", "mods", "('visual','textual')", lambda x: list(make_tuple(x)), lambda x: "-".join(m[:3] for m in x)),
            ("_loaders", "loaders", None, "('VisualAttribute','TextualAttribute')", lambda x: list(make_tuple(x)), None)
        ]
        self.autoset_params()
        self._run_mm_analysis = getattr(self._params.meta, "run_mm_analysis", False)

        if self._batch_size < 1:
            self._batch_size = self._data.transactions

        self._sampler = Sampler(self._data.i_train_dict, self._num_neg, self._seed)
        for m_id, m in enumerate(self._modalities):
            self.__setattr__(f'''_side_{m}''',
                             self._data.side_information.__getattribute__(f'''{self._loaders[m_id]}'''))

        all_multimodal_features = {m_id: self.__getattribute__(
                    f'''_side_{self._modalities[m_id]}''').object.get_all_features(item_order=self._data.items) for m_id, m in enumerate(self._modalities)}

        self._model = SiBraRModel(
                 # item multimodal features should also include interactions
                 num_users=self._num_users,
            num_items=self._num_items,
            lr=self._lr,
            input_dim=self._input_dim,
            w_decay=self._w_decay,
            cl_weight=self._cl_weight,
            cl_temp=self._cl_temp,
            cl_force_interactions_modality=self._cl_force_interactions_modality,
            cl_always_interactions_vs_agg=self._cl_always_interactions_vs_agg,
            agg_method=self._agg_method,
            mid_layers=self._mid_layers,
            emb_dim=self._emb_dim,
            b_norm=self._b_norm,
            sp_i_train_ratings=self._data.sp_i_train_ratings,
            modalities=self._modalities,
            u_prof=self._u_prof,
            dropout=self._dropout,
            norm_input_feat=self._norm_input_feat,
            norm_sbra_input=self._norm_sbra_input,
            item_multimodal_features=all_multimodal_features,  # dictionary of actual tensors
            random_seed=self._seed,
        )
        if hasattr(torch, 'compile') and not os.environ.get("ELLIOT_DETERMINISTIC"):
            self._model = torch.compile(self._model)

        self._missing_item_order = {}  # modality -> full shuffled item array (drawn once)
        self._missing_item_cache = {}  # (modality, proportion) -> sliced prefix of the above


    @property
    def name(self):
        return "SiBraR" \
               + f"_{self.get_base_params_shortcut()}" \
               + f"_{self.get_params_shortcut()}"

    def train(self):
        if self._restore:
            return self.restore_weights()

        start_epoch = self.load_resume_state() if self._warm_restart else 0
        for it in self.iterate(self._epochs, start=start_epoch):
            loss = 0
            steps = 0
            self._model.train()
            n_batch = int(self._data.transactions / self._batch_size) if self._data.transactions % self._batch_size == 0 else int(self._data.transactions / self._batch_size) + 1
            # self._data.edge_index = self._data.edge_index.sample(frac=1, replace=False).reset_index(drop=True)
            # edge_index = np.array([self._data.edge_index['userId'].tolist(), self._data.edge_index['itemId'].tolist()])
            with tqdm(total=n_batch, disable=not self._verbose) as t:
                for batch in prefetch_to_pinned(self._sampler.step(self._data.transactions, self._batch_size)):
                    steps += 1
                    device_batch = tuple(t.to(self._model.device, non_blocking=True)
                                         for t in batch)
                    self._model.optimizer.zero_grad()
                    current_loss = self._model.train_step(device_batch)
                    loss += current_loss

                    if math.isnan(loss) or math.isinf(loss) or (not loss):
                        break

                    t.set_postfix({'loss': f'{loss / steps:.5f}'})
                    t.update()
                # self._model.lr_scheduler.step()

            self.evaluate(it, loss / (it + 1))

        self._finalize_best()

        if self._warm_restart:
            self.clear_resume_state()

    def _recs_tsv_path(self):
        # Overrides RecMixin's default naming with the >240-char truncation.
        fold_index = self._fold_index if (hasattr(self, '_fold_index')
                                          and self._fold_index is not None) else None
        return os.path.abspath(os.sep.join([
            self._config.path_output_rec_result,
            build_recs_filename(self.name, fold_index),
        ]))

    def get_recommendations(self, k: int = 100):
        self._model.eval()
        with torch.no_grad():
            predictions_top_k_test = {}
            predictions_top_k_val = {}
            item_repr = self._model.get_item_representations(torch.arange(self._num_items).to(self._model.device))

            item_repr = self._model.aggregate(item_repr, dim=-2)
            for index, offset in enumerate(range(0, self._num_users, self._batch_size)):
                offset_stop = min(offset + self._batch_size, self._num_users)
                user_indices = torch.arange(index * self._batch_size, offset_stop).to(self._model.device)
                user_repr = self._model.get_user_representations(user_indices)
                predictions = self._model.predict(user_repr, item_repr)
                recs_val, recs_test = self.process_protocol(k, predictions, offset, offset_stop)
                predictions_top_k_val.update(recs_val)
                predictions_top_k_test.update(recs_test)
        return predictions_top_k_val, predictions_top_k_test

    def _get_missing_items(self, missing_modality: str, missing_proportion: float) -> np.ndarray:
        if missing_modality not in self._missing_item_order:
            rng = np.random.default_rng(self._seed)
            self._missing_item_order[missing_modality] = rng.permutation(self._num_items)
        key = (missing_modality, missing_proportion)
        if key not in self._missing_item_cache:
            num_missing = int(self._num_items * missing_proportion)
            self._missing_item_cache[key] = self._missing_item_order[missing_modality][:num_missing]
        return self._missing_item_cache[key]

    def get_recommendations_missing_modality(self, k: int = 100, missing_modality: str = 'emotion',
                                              missing_proportion: float = 0.5):
        if missing_modality not in self._modalities:
            raise ValueError(f"Modality '{missing_modality}' not in model modalities: {self._modalities}")
        missing_m_id = self._modalities.index(missing_modality)
        missing_item_ids = self._get_missing_items(missing_modality, missing_proportion)

        # availability[i, m] == True means item i has modality m; interactions branch is always available
        num_modalities_total = len(self._modalities) + 1
        availability = torch.ones(self._num_items, num_modalities_total, dtype=torch.bool)
        if len(missing_item_ids) > 0:
            availability[missing_item_ids, missing_m_id] = False

        self._model.eval()
        with torch.no_grad():
            all_items = torch.arange(self._num_items).to(self._model.device)
            item_repr_per_modality = self._model.get_item_representations(all_items)
            # [num_items, num_modalities+1, emb_dim]

            item_repr = self._model.aggregate(
                item_repr_per_modality, dim=-2, mask=availability.to(self._model.device)
            )  # [num_items, emb_dim]

            predictions_top_k_test = {}
            predictions_top_k_val = {}
            for index, offset in enumerate(range(0, self._num_users, self._batch_size)):
                offset_stop = min(offset + self._batch_size, self._num_users)
                user_indices = torch.arange(index * self._batch_size, offset_stop).to(self._model.device)
                user_repr = self._model.get_user_representations(user_indices)
                predictions = self._model.predict(user_repr, item_repr)
                recs_val, recs_test = self.process_protocol(k, predictions, offset, offset_stop)
                predictions_top_k_val.update(recs_val)
                predictions_top_k_test.update(recs_test)

        return predictions_top_k_val, predictions_top_k_test

    def evaluate_missing_modality(self, missing_modality: str = 'emotion', missing_proportion: float = 0.5):
        recs = self.get_recommendations_missing_modality(
            k=self.evaluator.get_needed_recommendations(),
            missing_modality=missing_modality,
            missing_proportion=missing_proportion,
        )
        result_dict = self.evaluator.eval(recs)
        self.logger.info(
            f'Missing modality — modality: {missing_modality}, proportion: {missing_proportion:.2f}, '
            f'val_{self._validation_metric}@{self._validation_k}: '
            f'{result_dict[self._validation_k]["val_results"][self._validation_metric]:.5f}'
        )
        return result_dict

    def get_recommendations_modality_subset(self, k: int = 100, modality_subset: tuple = None):
        """Evaluate using only a given subset of modalities for ALL items."""
        all_modalities = self._modalities + ['interactions']
        if modality_subset is None:
            modality_subset = tuple(all_modalities)

        # map modality names to indices in the per-modality dimension of get_item_representations
        mod_indices = []
        for m in modality_subset:
            if m == 'interactions':
                mod_indices.append(len(self._modalities))
            elif m in self._modalities:
                mod_indices.append(self._modalities.index(m))
            else:
                raise ValueError(f"Modality '{m}' not in {all_modalities}")

        self._model.eval()
        with torch.no_grad():
            all_items = torch.arange(self._num_items).to(self._model.device)
            item_repr_per_modality = self._model.get_item_representations(all_items)
            # [num_items, num_modalities+1, emb_dim]
            item_repr = self._model.aggregate(item_repr_per_modality[:, mod_indices, :], dim=-2)
            # [num_items, emb_dim]

            predictions_top_k_test = {}
            predictions_top_k_val = {}
            for index, offset in enumerate(range(0, self._num_users, self._batch_size)):
                offset_stop = min(offset + self._batch_size, self._num_users)
                user_indices = torch.arange(index * self._batch_size, offset_stop).to(self._model.device)
                user_repr = self._model.get_user_representations(user_indices)
                predictions = self._model.predict(user_repr, item_repr)
                recs_val, recs_test = self.process_protocol(k, predictions, offset, offset_stop)
                predictions_top_k_val.update(recs_val)
                predictions_top_k_test.update(recs_test)

        return predictions_top_k_val, predictions_top_k_test

    def run_missing_modality_analysis(self, k: int = 100, output_dir: str = None):
        """Evaluate all non-empty subsets of modalities (incl. interactions)."""
        all_modalities = self._modalities + ['interactions']
        results = {}

        subsets = []
        for r in range(1, len(all_modalities) + 1):
            subsets.extend(itertools.combinations(all_modalities, r))

        for subset in tqdm(subsets, desc='Missing modality analysis'):
            recs = self.get_recommendations_modality_subset(k=k, modality_subset=subset)
            result_dict = self.evaluator.eval(recs)
            results[subset] = result_dict
            val_score = result_dict[self._validation_k]["val_results"][self._validation_metric]
            self.logger.info(
                f'Missing modality analysis — subset: {subset}, '
                f'val_{self._validation_metric}@{self._validation_k}: {val_score:.5f}'
            )

        if output_dir is not None:
            import csv, json
            os.makedirs(output_dir, exist_ok=True)
            fold_str = f"_fold={self._fold_index}" if self._fold_index is not None else ""
            csv_path = os.path.join(output_dir, f"{self.name}_missing_modality{fold_str}.csv")
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                # header: subset, n_modalities, then all metrics at all cut-offs
                first = next(iter(results.values()))
                metric_cols = [
                    f"{split}_{metric}@{cutoff}"
                    for cutoff in sorted(first.keys())
                    for split in ("val", "test")
                    for metric in first[cutoff][f"{split}_results"].keys()
                ]
                writer.writerow(["subset", "n_modalities"] + metric_cols)
                for subset, rd in results.items():
                    row = ["|".join(subset), len(subset)]
                    for cutoff in sorted(rd.keys()):
                        for split in ("val", "test"):
                            for metric, v in rd[cutoff][f"{split}_results"].items():
                                row.append(round(v, 6))
                    writer.writerow(row)
            self.logger.info(f"Missing modality analysis saved to {csv_path}")

        return results

    def get_single_recommendation(self, mask, k, predictions, offset, offset_stop):
        self._model.eval()
        with torch.no_grad():
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
                fold_prefix = f"fold={self._fold_index} " if self._fold_index is not None else ""
                self.logger.info(
                    f'{fold_prefix}Epoch {(it + 1)}/{self._epochs} loss {loss / (it + 1):.5f} '
                    f'val_{self._validation_metric}@{self._validation_k} {val_metric:.5f}{delta_str}'
                )
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
                        }, self._get_saving_filepath())
                    else:
                        self.logger.warning("Saving weights FAILED. No model to save.")

                if self._save_recs and not (self._defer_recs and self._save_weights):
                    path = self._recs_tsv_path()
                    self.logger.info(f"Writing recommendations at: {path}")
                    store_recommendation(recs[1], path)

            if (self._warm_restart and it is not None
                    and not (it + 1) % self._resume_save_rate):
                self.save_resume_state(it)

    def restore_weights(self):
        try:
            checkpoint = torch.load(self._get_saving_filepath())
            self._model.load_state_dict(checkpoint['model_state_dict'])
            self._model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"Model correctly Restored")
            self.evaluate()
            if self._run_mm_analysis:
                output_dir = os.path.join(self._config.path_output_rec_performance, "missing_modality")
                self.run_missing_modality_analysis(
                    k=self.evaluator.get_needed_recommendations(),
                    output_dir=output_dir,
                )
            return True

        except Exception as ex:
            raise Exception(f"Error in model restoring operation! {ex}")

    