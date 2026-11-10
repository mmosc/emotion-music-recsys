import numpy as np
from scipy.spatial.distance import cdist, mahalanobis

_CALIBRATION_METRICS = ("mahalanobis", "energy")


class CalibratedReranking(object):
    """Greedy calibrated reranking (Steck, RecSys 2018) with a continuous distance."""

    def __init__(self, data, k, modalitiy, multimodal_feature, lam, calibration_metric="mahalanobis"):
        if calibration_metric not in _CALIBRATION_METRICS:
            raise ValueError(f"Unsupported calibration_metric: {calibration_metric!r} "
                              f"(expected one of {_CALIBRATION_METRICS})")

        self._data = data
        self._ratings = data.train_dict

        self._k = k
        self.multimodal_feature = multimodal_feature
        self.modalitiy = modalitiy
        self._lambda = lam
        self._calibration_metric = calibration_metric

        self._URM = self._data.sp_i_train

        self._users = self._data.users
        self._items = self._data.items
        self._private_users = self._data.private_users
        self._public_users = self._data.public_users
        self._private_items = self._data.private_items
        self._public_items = self._data.public_items

        if self._calibration_metric == "mahalanobis":
            covariance = np.cov(self.multimodal_feature.T)
            self._inv_covariance = np.linalg.pinv(covariance)
        else:
            self._energy_mu = self.multimodal_feature.mean(axis=0)
            sd = self.multimodal_feature.std(axis=0)
            self._energy_sd = np.where(sd < 1e-12, 1.0, sd)

    def _greedy_calibrate(self, user_feature, user_train_features, head_items_filtered, head_features, head_scores):
        if self._calibration_metric == "energy":
            return self._greedy_calibrate_energy(user_train_features, head_items_filtered, head_features, head_scores)
        return self._greedy_calibrate_mahalanobis(user_feature, head_items_filtered, head_features, head_scores)

    def _greedy_calibrate_mahalanobis(self, user_feature, head_items_filtered, head_features, head_scores):
        """Greedy relevance/calibration trade-off, vectorised over candidates."""
        features = np.asarray(head_features, dtype=float)
        scores = np.asarray(head_scores, dtype=float)
        user_feature = np.asarray(user_feature, dtype=float)
        n = len(head_items_filtered)

        # alive[i] tracks whether candidate i is still unselected, so the loop
        # never reallocates the feature/score arrays.
        alive = np.ones(n, dtype=bool)
        selected_sum = np.zeros(features.shape[1], dtype=float)
        selected_order = []

        for n_selected in range(n):
            idx = np.flatnonzero(alive)

            live_scores = scores[idx]
            min_score, max_score = live_scores.min(), live_scores.max()
            score_range = max_score - min_score
            if score_range > 0:
                rel_norm = (live_scores - min_score) / score_range
            else:
                rel_norm = np.ones_like(live_scores)

            profiles = (selected_sum + features[idx]) / (n_selected + 1)
            diffs = profiles - user_feature
            # einsum here is the batched form of d @ VI @ d for each row.
            calibration = np.sqrt(np.maximum(
                np.einsum("ij,jk,ik->i", diffs, self._inv_covariance, diffs), 0.0))
            # Match the scalar path, which skips the distance when the profile
            # coincides with the user profile and uses exactly 0.0.
            calibration[np.all(np.isclose(diffs, 0.0), axis=1)] = 0.0

            obj = (1 - self._lambda) * rel_norm - self._lambda * calibration

            # argmax takes the first maximum, matching the strict > comparison
            # in the original scan.
            chosen = idx[int(np.argmax(obj))]
            selected_order.append(head_items_filtered[chosen])
            selected_sum = selected_sum + features[chosen]
            alive[chosen] = False

        return selected_order

    def _greedy_calibrate_energy(self, user_train_features, head_items_filtered, head_features, head_scores):
        n_candidates = len(head_items_filtered)
        T = (np.asarray(user_train_features, dtype=float) - self._energy_mu) / self._energy_sd
        n = len(T)

        if n == 0:
            # no training history with features available -> fall back to plain relevance order
            order = sorted(range(n_candidates), key=lambda i: head_scores[i], reverse=True)
            return [head_items_filtered[i] for i in order]

        cand_features = (np.asarray(head_features, dtype=float) - self._energy_mu) / self._energy_sd
        d_tt_mean = cdist(T, T).mean() if n > 1 else 0.0
        d_to_T = cdist(cand_features, T).mean(axis=1)   # per-candidate mean distance to T, computed once
        d_between = cdist(cand_features, cand_features)  # pairwise candidate distances, computed once

        min_score, max_score = min(head_scores), max(head_scores)
        score_range = max_score - min_score

        remaining = set(range(n_candidates))
        selected = []
        cross_sum = 0.0   # sum over selected candidates i of n * d_to_T[i]
        within_sum = 0.0

        while remaining:
            best_idx, best_obj = None, -np.inf
            best_cross, best_within = cross_sum, within_sum
            m = len(selected) + 1
            for idx in remaining:
                new_cross_sum = cross_sum + n * d_to_T[idx]
                new_within_sum = within_sum + 2.0 * sum(d_between[idx, s] for s in selected)
                d_tr = new_cross_sum / (n * m)
                d_rr = new_within_sum / (m * m)
                calibration = max(2.0 * d_tr - d_tt_mean - d_rr, 0.0)

                rel_norm = (head_scores[idx] - min_score) / score_range if score_range > 0 else 1.0
                obj = (1 - self._lambda) * rel_norm - self._lambda * calibration

                if obj > best_obj:
                    best_obj = obj
                    best_idx = idx
                    best_cross, best_within = new_cross_sum, new_within_sum

            selected.append(best_idx)
            remaining.discard(best_idx)
            cross_sum, within_sum = best_cross, best_within

        return [head_items_filtered[i] for i in selected]

    def rerank_recommendations(self, recs):
        reranked_recs = {}
        for u, user_recs in recs.items():
            user_id = self._public_users.get(u)

            # only the top-k candidates get reranked; everything beyond k stays untouched.
            head = user_recs[:self._k]
            tail = user_recs[self._k:]

            head_items = [rec for rec, _ in head]
            head_scores = [score for _, score in head]
            head_items_filtered = []
            head_scores_filtered = []
            for item, score in zip(head_items, head_scores):
                if self._public_items.get(item) is not None:
                    head_items_filtered.append(item)
                    head_scores_filtered.append(score)
            head_public_items = [self._public_items.get(rec) for rec in head_items_filtered]

            user_items = self._URM[user_id].indices

            user_train_features = self.multimodal_feature[user_items]
            user_feature = np.mean(user_train_features, axis=0)
            head_features = list(self.multimodal_feature[head_public_items])

            reranked_head_items = self._greedy_calibrate(
                user_feature, user_train_features, head_items_filtered, head_features, head_scores_filtered
            )

            reranked_head = [(item, score) for item, score in zip(
                reranked_head_items,
                np.linspace(1.0, 0.0, num=max(len(reranked_head_items), 1))
            )]
            reranked_recs[u] = reranked_head + tail

        return reranked_recs
