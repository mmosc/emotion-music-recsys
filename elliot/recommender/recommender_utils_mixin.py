import os

import numpy as np
from tqdm import tqdm

from elliot.utils.write import store_recommendation

import wandb


class RecMixin(object):

    def train(self):
        if self._restore:
            return self.restore_weights()

        for it in range(self._num_iters):
            loss = 0
            steps = 0
            with tqdm(total=int(self._data.transactions // self._batch_size), disable=not self._verbose) as t:
                for batch in self._sampler.step(self._data.transactions, self._batch_size):
                    steps += 1
                    loss += self._model.train_step(batch).numpy()
                    t.set_postfix({'loss': f'{loss/steps:.5f}'})
                    t.update()

            self.evaluate(it, loss/(it + 1))

    def evaluate(self, it=None, loss=0, wandb_log=False):
        if (it is None) or (not (it + 1) % self._validation_rate):
            recs = self.get_recommendations(self.evaluator.get_needed_recommendations())
            result_dict = self.evaluator.eval(recs)

            self._losses.append(loss)

            self._results.append(result_dict)

            if it is None and wandb_log:
                wandb.log(
                    {
                        # For now, we only log one metric
                        "val_ndcg5": result_dict[5]['val_results']['nDCG'],
                        "test_ndcg5": result_dict[5]['test_results']['nDCG'],
                    },
                )

            if it is not None:
                self.logger.info(f'Epoch {(it + 1)}/{self._epochs} loss {loss/(it + 1):.5f}')
                wandb.log(
                    {
                        "epochs": it + 1,
                        # for now it does not allow logging the two losses separately
                        "loss": loss / (it + 1),
                        # For now, we only log one metric
                        "val_ndcg5": result_dict[5]['val_results']['nDCG'],
                        "test_ndcg5": result_dict[5]['test_results']['nDCG'],
                    },
                )
            else:
                self.logger.info(f'Finished')

            if self._save_recs:
                self.logger.info(f"Writing recommendations at: {self._config.path_output_rec_result}")
                fold_str = f"_fold={self._fold_index}" if (hasattr(self, '_fold_index') and self._fold_index is not None) else ""
                if it is not None:
                    store_recommendation(recs[1], os.path.abspath(
                        os.sep.join([
                            self._config.path_output_rec_result,
                            f"{self.name}{fold_str}_it={it + 1}.tsv"
                        ])
                    ))
                else:
                    store_recommendation(recs[1], os.path.abspath(
                        os.sep.join([
                            self._config.path_output_rec_result,
                            f"{self.name}{fold_str}.tsv"
                        ])
                    ))

            if (len(self._results) - 1) == self.get_best_arg():
                if it is not None:
                    self._params.best_iteration = it + 1
                self.logger.info("******************************************")
                self.best_metric_value = self._results[-1][self._validation_k]["val_results"][self._validation_metric]
                if self._save_weights:
                    if hasattr(self, "_model"):
                        self._model.save_weights(self._saving_filepath)
                    else:
                        self.logger.warning("Saving weights FAILED. No model to save.")



    def get_recommendations(self, k: int = 100):
        predictions_top_k_test = {}
        predictions_top_k_val = {}
        for index, offset in enumerate(range(0, self._num_users, self._batch_size)):
            offset_stop = min(offset + self._batch_size, self._num_users)
            predictions = self._model.predict(self._data.sp_i_train[offset:offset_stop].toarray())
            recs_val, recs_test = self.process_protocol(k, predictions, offset, offset_stop)
            predictions_top_k_val.update(recs_val)
            predictions_top_k_test.update(recs_test)

        return predictions_top_k_val, predictions_top_k_test

    def process_protocol(self, k, *args):

        if not self._negative_sampling:
            recs = self.get_single_recommendation(self.get_candidate_mask(), k, *args)
            return recs, recs
        else:
            return self.get_single_recommendation(self.get_candidate_mask(validation=True), k, *args) if hasattr(self._data, "val_dict") else {}, \
                   self.get_single_recommendation(self.get_candidate_mask(), k, *args)

    def get_single_recommendation(self, mask, k, predictions, offset, offset_stop):
        v, i = self._model.get_top_k(predictions, mask[offset: offset_stop], k=k)
        items_ratings_pair = [list(zip(map(self._data.private_items.get, u_list[0]), u_list[1]))
                              for u_list in list(zip(i.numpy(), v.numpy()))]
        return dict(zip(map(self._data.private_users.get, range(offset, offset_stop)), items_ratings_pair))

    def restore_weights(self):
        try:
            self._model.load_weights(self._saving_filepath)
            print(f"Model correctly Restored")
            self.evaluate()
            return True

        except Exception as ex:
            raise Exception(f"Error in model restoring operation! {ex}")

        return False

    def _recs_tsv_path(self):
        """Path of the recs TSV for this (model, fold). Default naming shared
        by DeepMF/LightGCN; models with their own conventions (SiBraR's
        240-char truncation, LightGCNM's legacy fold-less name) override."""
        fold_str = f"_fold={self._fold_index}" if (hasattr(self, '_fold_index')
                                                   and self._fold_index is not None) else ""
        return os.path.abspath(os.sep.join([
            self._config.path_output_rec_result, f"{self.name}{fold_str}.tsv"]))

    def _finalize_best(self):
        """Shared finalize step for the deferred meta flags: reload the best
        weights saved by evaluate()'s best-epoch branch (all models using
        this save the same {'model_state_dict', 'optimizer_state_dict'}
        checkpoint), recompute the recommendations once, then
        (meta.defer_recs) write the same TSV the best-epoch branch would
        have written on every improvement epoch and/or
        (meta.defer_test_eval) compute the test metrics skipped during
        training and patch them into the best epoch's result entry, so
        get_results() -> intermediate_fold_*.json see real values.
        state_dict save/load is exact and the forward shapes are unchanged,
        so both outputs are bit-identical to computing them in-loop."""
        import torch
        needs_recs = self._save_recs and self._defer_recs and self._save_weights
        needs_test = self._defer_test_eval and self._save_weights
        if not (needs_recs or needs_test) or not self._results:
            return
        try:
            checkpoint = torch.load(self._get_saving_filepath())
            self._model.load_state_dict(checkpoint['model_state_dict'])
        except Exception as ex:
            self.logger.warning(f"finalize: could not reload best weights "
                                f"({ex}) — deferred outputs not produced")
            return
        recs = self.get_recommendations(self.evaluator.get_needed_recommendations())

        if needs_test:
            test_result = self.evaluator.eval(recs, sides=("test",))
            best = self._results[self.get_best_arg()]
            for k, result in test_result.items():
                best[k]["test_results"] = result["test_results"]
                best[k]["test_statistical_results"] = result["test_statistical_results"]

        if needs_recs:
            path = self._recs_tsv_path()
            self.logger.info(f"Writing recommendations at: {path}")
            store_recommendation(recs[1], path)

    def _resume_state_path(self):
        return self._saving_filepath + "_resume"

    def save_resume_state(self, it):
        """Persist everything needed to continue training after this
        validation epoch as if never interrupted: model + optimizer state and
        the loss/results histories that early stopping and best-epoch
        selection read. Written at every validation epoch (unlike the
        best-weights file, which only reflects the best epoch so far)."""
        import torch
        torch.save({
            'epoch': it + 1,
            'model_state_dict': self._model.state_dict(),
            'optimizer_state_dict': self._model.optimizer.state_dict(),
            'losses': self._losses,
            'results': self._results,
            'best_iteration': getattr(self._params, 'best_iteration', None),
            'best_metric_value': self.best_metric_value,
        }, self._resume_state_path())

    def load_resume_state(self):
        """Restore the state written by save_resume_state, returning the
        epoch to continue from (0 when no usable state exists). map_location
        matters: an OOM-requeued fold relaunches on a different GPU than the
        one the state was saved from."""
        import torch
        path = self._resume_state_path()
        if not os.path.exists(path):
            self.logger.info(f"No resume state at {path} — training from scratch")
            return 0
        try:
            device = next(self._model.parameters()).device
            state = torch.load(path, map_location=device)
            self._model.load_state_dict(state['model_state_dict'])
            self._model.optimizer.load_state_dict(state['optimizer_state_dict'])
        except Exception as ex:
            self.logger.warning(f"Could not load resume state {path} ({ex}) — training from scratch")
            return 0
        self._losses = state['losses']
        self._results = state['results']
        if state['best_iteration'] is not None:
            self._params.best_iteration = state['best_iteration']
        self.best_metric_value = state['best_metric_value']
        self.logger.info(f"Resuming training from epoch {state['epoch'] + 1} (resume state {path})")
        return state['epoch']

    def clear_resume_state(self):
        path = self._resume_state_path()
        if os.path.exists(path):
            os.remove(path)

    def get_candidate_mask(self, validation=False):
        if self._negative_sampling:
            if validation:
                return self._data.val_mask
            else:
                return self._data.test_mask
        else:
            return self._data.allunrated_mask

    def get_loss(self):
        if self._optimize_internal_loss:
            return min(self._losses)
        else:
            return -max([r[self._validation_k]["val_results"][self._validation_metric] for r in self._results])

    def get_params(self):
        return self._params.__dict__

    def get_results(self):
        return self._results[self.get_best_arg()]

    def get_best_arg(self):
        if self._optimize_internal_loss:
            val_results = np.argmin(self._losses)
        else:
            val_results = np.argmax([r[self._validation_k]["val_results"][self._validation_metric] for r in self._results])
        return val_results

    def iterate(self, epochs, start=0):
        for iteration in range(start, epochs):
            if self._early_stopping.stop(self._losses[:], self._results):
                self.logger.info(f"Met Early Stopping conditions: {self._early_stopping}")
                break
            else:
                yield iteration
