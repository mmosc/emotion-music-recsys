from abc import ABC

import torch
import numpy as np
import random
from torch import nn


class DeepMFModel(torch.nn.Module, ABC):
    def __init__(self,
                 num_users,
                 num_items,
                 embedding_dim,
                 user_mlp,
                 item_mlp,
                 reg,
                 similarity,
                 max_ratings,
                 sp_i_train_ratings,
                 learning_rate=0.01,
                 # from equation (13) of the original paper
                 mu=1.e-6,
                 random_seed=42,
                 name="DeepMF",
                 **kwargs):
        super().__init__(**kwargs)
        # set seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = True
        #torch.use_deterministic_algorithms(True)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.user_mlp = list(user_mlp)
        self.item_mlp = list(item_mlp)
        self.reg = reg
        self.similarity = similarity
        self.max_ratings = max_ratings
        self._sp_i_train_ratings = sp_i_train_ratings
        self.mu = mu
        # User and item profiles (rows and cols in the interaction matrix)
        # as pre-trained embeddings
        self.user_embedding = torch.nn.Embedding.from_pretrained(
                        torch.tensor(self._sp_i_train_ratings.toarray(), dtype=torch.float32, device=self.device), freeze=True)

        self.item_embedding = torch.nn.Embedding.from_pretrained(
            torch.tensor(self._sp_i_train_ratings.T.toarray(), dtype=torch.float32, device=self.device), freeze=True)

        layers = []
        user_dims = [self.num_items] + self.user_mlp + [self.embedding_dim]
        for i, (d1, d2) in enumerate(zip(user_dims[:-1], user_dims[1:])):
            layer = nn.Linear(in_features=d1, out_features=d2)
            torch.nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

            layer.to(self.device)
            layers.append(layer)
            if layers != len(user_dims) - 2:
                activation = nn.ReLU().to(self.device)
                layers.append(activation)
        self.user_embedding_module = torch.nn.Sequential(*layers)

        for user_param in self.user_embedding_module.parameters():
            user_param.requires_grad = True

        layers = []
        item_dims = [self.num_users] + self.item_mlp + [self.embedding_dim]
        for i, (d1, d2) in enumerate(zip(item_dims[:-1], item_dims[1:])):
            layer = nn.Linear(in_features=d1, out_features=d2)
            torch.nn.init.xavier_uniform_(layer.weight)
            layer.to(self.device)
            layers.append(layer)
            if layers != len(item_dims) - 2:
                activation = nn.ReLU().to(self.device)
                layers.append(activation)

        self.item_embedding_module = torch.nn.Sequential(*layers)
        for item_param in self.item_embedding_module.parameters():
            item_param.requires_grad = True

        self.cosine_func = nn.CosineSimilarity(dim=-1)
        # self.cosine_func = self._cosine
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=self.reg)
        self.name = 'DeepMatrixFactorization'

    def _cosine(self, layer_0, layer_1):
        """Cosine similarity between two tensors."""
        layer_0_norm = torch.nn.functional.normalize(layer_0, p=2, dim=-1)
        layer_1_norm = torch.nn.functional.normalize(layer_1, p=2, dim=-1)
        return torch.matmul(layer_0_norm, layer_1_norm.T)

    def get_user_representations(self, users):
        feature = self.user_embedding(users)
        feature = self.user_embedding_module(feature)
        return feature

    def get_item_representations(self, items):
        feature = self.item_embedding(items)
        feature = self.item_embedding_module(feature)
        return feature

    def forward(self, inputs, **kwargs):
        users, items = inputs


        # this should be used in training
        if isinstance(users, np.ndarray):
            user_tensor = torch.from_numpy(users).to(self.device)
        else:
            user_tensor = users
        if isinstance(items, np.ndarray):
            items_tensor = torch.from_numpy(items).to(self.device)
        else:
            items_tensor = items

        u_repr = self.get_user_representations(user_tensor)
        i_repr = self.get_item_representations(items_tensor)
        return u_repr, i_repr

    def forward_eval(self, inputs, **kwargs):

        user_tensor, items_tensor = inputs

        u_repr = self.get_user_representations(user_tensor)
        i_repr = self.get_item_representations(items_tensor)
        return u_repr, i_repr

    def predict(self, start_user, stop_user, **kwargs):
        """Predict scores for users in the range [start_user, stop_user) over all items.
        """
        user = torch.arange(start_user, stop_user).to(self.device)
        item = torch.arange(self.num_items).to(self.device)
        user_repr, item_repr = self.forward_eval(inputs=(user, item))
        user_repr, item_repr = user_repr.to(self.device), item_repr.to(self.device)
        if self.similarity == 'cosine':
            preds = self._cosine(user_repr, item_repr)
            # print("[   PREDS SHAPE IN PREDICT        ]: ", preds.shape)
            # preds[preds < self.mu] = self.mu
        else:
            preds = torch.mm(user_repr, item_repr.T)
            preds = torch.sigmoid(preds)
        return preds

    def train_step(self, batch):
        user, item, label = batch

        label = torch.FloatTensor(label).to(self.device)

        label = label / self.max_ratings
        self.optimizer.zero_grad()
        # print(label[:20])
        user_repr, item_repr = self.forward(inputs=(user, item))

        # print(f"user_repr shape: {user_repr.shape}")
        # print(f"item_repr shape: {item_repr.shape}")

        if self.similarity == 'cosine':
            # print("[   USER REPR SHAPE IN TRAIN      ]: ", user_repr.shape)
            # print("[   ITEM REPR SHAPE IN TRAIN      ]: ", item_repr.shape)
            preds = self.cosine_func(user_repr, item_repr)
            preds[preds < self.mu] = self.mu
            # print(preds.shape)
        else:
            preds = torch.mm(user_repr, item_repr.T)
            preds = torch.sigmoid(preds)

        # print(f"preds shape: {preds.shape}")
        loss = nn.BCELoss()(preds, label)
        # print(f"loss shape: {loss.shape}")

        loss.backward()
        self.optimizer.step()
        return loss.detach().cpu().numpy()


    def get_top_k(self, preds, train_mask, k=100):
        return torch.topk(torch.where(torch.tensor(train_mask).to(self.device), preds.to(self.device),
                                      torch.tensor(-np.inf).to(self.device)), k=k, sorted=True)

