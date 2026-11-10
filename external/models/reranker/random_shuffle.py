import numpy as np


class RandomShuffle:
    """Null control for the emotion reranker."""

    def __init__(self, data, k, seed=42):
        self._data = data
        self._k = k
        self._seed = seed
        self._public_items = data.public_items

    def rerank_recommendations(self, recs, fold=0):
        reranked_recs = {}
        rng = np.random.default_rng([self._seed, fold])

        for u, user_recs in recs.items():
            head = user_recs[:self._k]
            tail = user_recs[self._k:]

            # Same id filtering Similarity applies, so both rerankers see an
            # identical candidate set and differ only in ordering.
            head_items = [rec for rec, _ in head if self._public_items.get(rec) is not None]

            order = rng.permutation(len(head_items))
            shuffled = [head_items[i] for i in order]

            scores = np.arange(len(shuffled), 0, -1, dtype=float)
            reranked_recs[u] = list(zip(shuffled, scores)) + tail

        return reranked_recs
