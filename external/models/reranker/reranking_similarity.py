import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances


class Similarity(object):

    def __init__(self, data, similarity, k, modalitiy, multimodal_feature):
        self._data = data
        self._ratings = data.train_dict

        self._similarity = similarity
        self._k = k
        self.multimodal_feature = multimodal_feature
        self.modalitiy = modalitiy

        self._URM = self._data.sp_i_train

        self._users = self._data.users
        self._items = self._data.items
        self._private_users = self._data.private_users
        self._public_users = self._data.public_users
        self._private_items = self._data.private_items
        self._public_items = self._data.public_items

    def rerank_recommendations(self, recs):
        reranked_recs = {}
        for u, user_recs in recs.items():
            user_id = self._public_users.get(u)

            # only the top-k candidates get reranked; everything beyond k stays untouched.
            head = user_recs[:self._k]
            tail = user_recs[self._k:]

            head_items = [rec for rec, _ in head]
            head_items_filtered = [rec for rec in head_items if self._public_items.get(rec) is not None]
            head_public_items = [self._public_items.get(rec) for rec in head_items_filtered]

            user_items = self._URM[user_id].indices

            user_feature = np.mean(self.multimodal_feature[user_items], axis=0)
            head_features = self.multimodal_feature[head_public_items]

            if self._similarity == "cosine":
                # higher cosine similarity = more similar -> sort descending
                scores = cosine_similarity(user_feature.reshape(1, -1), head_features).flatten()
                reverse = True
            elif self._similarity == "euclidean":
                # lower euclidean distance = more similar -> sort ascending
                scores = euclidean_distances(user_feature.reshape(1, -1), head_features).flatten()
                reverse = False
            else:
                raise ValueError(f"Unsupported reranking similarity metric: {self._similarity}")

            reranked_head = sorted(zip(head_items_filtered, scores), key=lambda x: x[1], reverse=reverse)
            reranked_recs[u] = reranked_head + tail

        return reranked_recs