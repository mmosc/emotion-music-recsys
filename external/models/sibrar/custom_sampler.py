import numpy as np

class Sampler:
    def __init__(self, indexed_ratings, num_neg=128, seed=42):
        self._rng = np.random.RandomState(seed)
        self._indexed_ratings = indexed_ratings
        self._users = list(self._indexed_ratings.keys())
        self._nusers = len(self._users)
        self._users_arr = np.array(self._users)
        self._items = list({k for a in self._indexed_ratings.values() for k in a.keys()})
        self._nitems = len(self._items)
        self._ui_dict = {u: np.array(list(set(indexed_ratings[u]))) for u in indexed_ratings}
        self._lui_dict = {u: len(v) for u, v in self._ui_dict.items()}
        self.num_neg = num_neg

        all_items = np.array(self._items)

        # Positive pools: flat array + per-user sizes and offsets for vectorised lookup
        pos_pools = [self._ui_dict[u] for u in self._users]
        self._lui_arr = np.array([len(p) for p in pos_pools], dtype=np.int64)
        self._ui_offsets = np.concatenate([[0], np.cumsum(self._lui_arr[:-1])])
        self._ui_flat = np.concatenate(pos_pools)

        # Negative pools: flat array + per-user sizes and offsets for vectorised lookup
        # (replaces the per-user _neg_pool dict)
        neg_pools = [np.setdiff1d(all_items, self._ui_dict[u], assume_unique=True)
                     for u in self._users]
        self._pool_sizes = np.array([len(p) for p in neg_pools], dtype=np.int64)
        self._pool_offsets = np.concatenate([[0], np.cumsum(self._pool_sizes[:-1])])
        self._neg_pool_flat = np.concatenate(neg_pools)

    def step(self, events: int, batch_size: int):
        nusers = self._nusers
        num_neg = self.num_neg

        for batch_start in range(0, events, batch_size):
            bs = min(batch_size, events - batch_start)

            u_idx = self._rng.randint(nusers, size=bs)

            # Positive: one item per user sampled uniformly from their history
            pos_local = (self._rng.rand(bs) * self._lui_arr[u_idx]).astype(np.int64)
            pos_items = self._ui_flat[self._ui_offsets[u_idx] + pos_local]

            neg_local = (self._rng.rand(bs, num_neg)
                         * self._pool_sizes[u_idx][:, None]).astype(np.int64)
            neg_items = self._neg_pool_flat[self._pool_offsets[u_idx][:, None] + neg_local]

            yield self._users_arr[u_idx][:, None], pos_items[:, None], neg_items[:, None]
