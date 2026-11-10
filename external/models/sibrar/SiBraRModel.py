from abc import ABC

import os

import torch
import numpy as np
import random
from torch import nn
import torch.nn.functional as F
import collections

class SparseEmbedding(nn.Module):
    def __init__(self, sparse_matrix, device):
        super().__init__()
        self.sparse_matrix = sparse_matrix  # Keep on CPU as scipy sparse
        self.device = device

    def forward(self, indices):
        # Fetch only the rows we need (indices are user/item IDs)
        # Save original shape to restore later
        original_shape = indices.shape

        # Convert indices to numpy and flatten for indexing
        idx = indices.cpu().numpy().flatten()

        # Index sparse matrix and convert to dense
        batch_data = torch.tensor(
            self.sparse_matrix[idx].toarray(),
            dtype=torch.float32,
            device=self.device
        )

        feature_dim = batch_data.shape[-1]
        batch_data = batch_data.reshape(*original_shape, feature_dim)

        return batch_data


class MyBatchNorm(nn.Module):
    def __init__(self, num_features):
        super(MyBatchNorm, self).__init__()
        self.num_features = num_features
        self.batch_norm = nn.BatchNorm1d(num_features)

    def forward(self, x):
        return self.batch_norm(x.flatten(0, -2)).reshape(x.shape)

class L2NormalizationLayer(nn.Module):
    def __init__(self, dim=-1, eps=1e-12):
        super(L2NormalizationLayer, self).__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x):
        return F.normalize(x, p=2, dim=self.dim, eps=self.eps)


class SiBraRModel(torch.nn.Module, ABC):
    def __init__(self,
                 num_users,
                 num_items,
                 lr,
                 input_dim,
                 mid_layers,
                 emb_dim,
                 b_norm,
                 w_decay,
                 cl_weight,
                 cl_temp,
                 sp_i_train_ratings,
                 modalities, # list of strings
                 u_prof,
                 dropout,
                 norm_input_feat,
                 norm_sbra_input,
                 # item multimodal features should also include interactions
                 item_multimodal_features, # actual tensors
                 random_seed,
                 cl_force_interactions_modality=False,
                 cl_always_interactions_vs_agg=False,
                 agg_method='mean',
                 name="ItemSiBraR",
                 **kwargs
                 ):
        super().__init__()

        # set seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = True
        if os.environ.get("ELLIOT_DETERMINISTIC"):
            torch.use_deterministic_algorithms(True)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.num_users = num_users
        self.num_items = num_items
        self.input_dim = input_dim
        self.mid_layers = mid_layers
        self.emb_dim = emb_dim
        self.b_norm = b_norm
        self.lr = lr
        self.w_decay = w_decay
        self.cl_weight = cl_weight
        self.cl_temp = cl_temp
        self._sp_i_train_ratings = sp_i_train_ratings
        self.modalities = modalities
        self.name = name
        self.u_prof = u_prof
        self.dropout = dropout
        self.norm_input_feat = norm_input_feat
        self.norm_sbra_input = norm_sbra_input
        self.cl_force_interactions_modality = cl_force_interactions_modality
        self.cl_always_interactions_vs_agg = cl_always_interactions_vs_agg
        self.agg_method = agg_method

        assert not (self.cl_force_interactions_modality and self.cl_always_interactions_vs_agg), \
            "cl_force_interactions_modality and cl_always_interactions_vs_agg are mutually exclusive"

        self.modality_projection_layers = {}
        self.item_multimodal_features = item_multimodal_features

        item_embedding_modules = {}

        for m_id, m in enumerate(self.modalities):
            layers = [(f'{m}_as_embedding', torch.nn.Embedding.from_pretrained(
                        torch.tensor(self.item_multimodal_features[m_id], dtype=torch.float32, device=self.device)
                    ))]
            # Original SiBraR uses L1 regularization and this is where weight decay is used
            #     layers.append((f'{m}_norm_layer', L1(dim=1)))
            if self.norm_input_feat:
                layers.append((f'{m}_norm_layer', L2NormalizationLayer(dim=-1)))

            layers.append((f'{m}_projector', torch.nn.Linear(self.item_multimodal_features[m_id].shape[1], self.input_dim).to(self.device)))
            # original sibrar does not use activation after projection
            # layers.append((f'{m}_projector_activation', nn.ReLU()))
            layers = collections.OrderedDict(layers)

            item_embedding_modules[str(m_id)] = torch.nn.Sequential(layers)


        layers = [(f'profile_as_embedding', SparseEmbedding(self._sp_i_train_ratings.T, device=self.device)
        # torch.nn.Embedding.from_pretrained(
        #             torch.tensor(self._sp_i_train_ratings.T.toarray(), dtype=torch.float32, device=self.device))
                )]
        if self.norm_input_feat:
            layers.append((f'profile_norm_layer', L2NormalizationLayer(dim=-1)))
        layers.append((f'profile_projector', torch.nn.Linear(self.num_users, self.input_dim).to(self.device)))
        # original sibrar does not use activation after projection
        # layers.append((f'profile_projector_activation', nn.ReLU()))

        layers = collections.OrderedDict(layers)
        item_embedding_modules[str(len(self.modalities))] = torch.nn.Sequential(layers)

        self.item_embedding_modules = nn.ModuleDict(item_embedding_modules)

        # this is the actual single branch, shared by all item modalities
        single_branch_layers_dim = [self.input_dim] + self.mid_layers + [self.emb_dim]

        layers = collections.OrderedDict()
        if self.dropout > 0.:
            layers[f'single_branch_dropout_input'] = nn.Dropout(p=self.dropout)

        total_iterations = len(single_branch_layers_dim[:-1])
        for i, (d1, d2) in enumerate(zip(single_branch_layers_dim[:-1], single_branch_layers_dim[1:])):
            layers[f"single_branch_layer_{i}"] = nn.Linear(in_features=d1, out_features=d2).to(self.device)

            # apply batch normalization before activation fn (see http://torch.ch/blog/2016/02/04/resnets.html)
            # +1 to not immediately put normalization after first layer if 'apply_b_norm_e' >= 2

            if b_norm:
                # see https://discuss.pytorch.org/t/problems-using-batchnorm1d/14730/5
                # batchnorm1d(x.flatten(0, -2)).reshape(x.shape)
                layers[f"batch_norm_{i}"] = MyBatchNorm(num_features=d2).to(self.device)

            if i < total_iterations - 1:
                # only add activation functions in intermediate layers
                layers[f"single_branch_relu_{i}"] = nn.ReLU().to(self.device)

            if self.dropout > 0.:
                layers[f'single_branch_dropout_{i}'] = nn.Dropout(p=self.dropout)

        # if b_norm_e == -1:
        #     layers[f"final_batch_norm"] = nn.BatchNorm1d(num_features=single_branch_layers_dim[-1])

        # Final ReLU activation
        layers[f"final_activation"] = nn.ReLU()

        self.single_branch = torch.nn.Sequential(layers)

        # Right now only supports interactions, only, for users
        if self.u_prof:
            self.user_embedding_module = torch.nn.Sequential(
                collections.OrderedDict([
                ('profile_as_embedding', SparseEmbedding(self._sp_i_train_ratings, device=self.device)
                # torch.nn.Embedding.from_pretrained(
                #         torch.tensor(self._sp_i_train_ratings.toarray(), dtype=torch.float32, device=self.device))
                        ),
                ('profile_projector', torch.nn.Linear(self.num_items, self.emb_dim).to(self.device)),
                ('profile_projector_relu', nn.ReLU().to(self.device)),
            ]))
        else:
            self.user_embedding_module = torch.nn.Embedding(self.num_users, self.emb_dim).to(self.device)

        if self.agg_method == 'gate':
            # item-only gate: scores each modality slot from its own embedding,
            # so gate weights for an item don't depend on which user is being scored
            self.gate_module = torch.nn.Linear(self.emb_dim, 1).to(self.device)

        # AdamW is what was used for SiBraR
        # with weight decay instead of l2 reg
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.w_decay)


    def get_user_representations(self, users):
        feature = self.user_embedding_module(users)
        return feature


    def get_item_representations(self, items):        
        features = torch.zeros((*items.squeeze().shape, len(self.modalities) + 1, self.emb_dim)).to(self.device)
        for m_id, m in enumerate(self.modalities):
            feature = self.item_embedding_modules[str(m_id)](items)
            # Move to the layers?
            if self.norm_sbra_input:
                feature = nn.functional.normalize(feature, p=2, dim=-1)
            feature = self.single_branch(feature)
            features[..., m_id, :] = feature.squeeze()

        # Interactions
        m_id = len(self.modalities)
        feature = self.item_embedding_modules[str(m_id)](items)
        id_mean = feature.abs().mean().item()

        if self.norm_sbra_input:
            feature = nn.functional.normalize(feature, p=2, dim=-1)
        feature = self.single_branch(feature)
        features[..., m_id, :] = feature.squeeze()

        # avg_activity = features.abs().mean(dim=(0, 2)).detach().cpu().numpy()

        return features # Shape is [batch_size, num_modalities, embedding_dimension]

    def _aggregate(self, modality_reps, dim=-2, mask=None):
        # modality_reps: [..., num_slots, emb_dim]; mask: optional bool [..., num_slots], True = available
        if self.agg_method == 'mean':
            if mask is None:
                return torch.mean(modality_reps, dim=dim)
            mask_f = mask.unsqueeze(-1).float()
            count = mask_f.sum(dim=dim).clamp(min=1.0)
            return (modality_reps * mask_f).sum(dim=dim) / count

        elif self.agg_method == 'gate':
            gate_logits = self.gate_module(modality_reps).squeeze(-1)  # [..., num_slots]
            if mask is not None:
                gate_logits = gate_logits.masked_fill(~mask, float('-inf'))
            gate_weights = torch.softmax(gate_logits, dim=-1)
            return (modality_reps * gate_weights.unsqueeze(-1)).sum(dim=dim)

        else:
            raise ValueError(f"Unknown agg_method: {self.agg_method}")

    def aggregate(self, modality_reps, dim=-2, mask=None):
        return self._aggregate(modality_reps, dim=dim, mask=mask)

    def forward(self, inputs, **kwargs):
        # gu and gi should be the user and item batches
        users, items = inputs

        user_tensor = torch.tensor(users).to(self.device)
        items_tensor = torch.tensor(items).to(self.device)

        u_repr = self.get_user_representations(user_tensor)
        i_repr = self.get_item_representations(items_tensor)
        # i_repr = torch.mean(i_repr, dim=-2)
        return u_repr, i_repr

    def loss_contrastive_all_items(self, contrastive_modality_reps):
        # shape is [num_users, 1 + n_negs, 2, embedding_dim]
        contrastive_modality_reps = contrastive_modality_reps.reshape(-1, contrastive_modality_reps.size(-2), contrastive_modality_reps.size(-1))
        logits = contrastive_modality_reps[:, 0, :] @ contrastive_modality_reps[:, 1, :].transpose(-2, -1) / self.cl_temp

            


        # Positive keys are the entries on the diagonal, therefore these are the correct labels:
        # [batch_size, 0, 1, ..., batch_size - 1]
        labels = torch.arange(logits.shape[-1], device=contrastive_modality_reps.device)#.repeat(1, *logits.shape[:-2], 1)

        # Logits change depending on which modality we are "retrieving"
        logits_c_to_p = logits.reshape(-1, logits.shape[-1])
        logits_p_to_c = logits.transpose(-2, -1).reshape(-1, logits.shape[-1])

        # labels = labels.reshape(-1)
        x_y_loss = F.cross_entropy(logits_c_to_p, labels)
        y_x_loss = F.cross_entropy(logits_p_to_c, labels)

        contrastive_loss = x_y_loss + y_x_loss
        return contrastive_loss

    def loss_contrastive(self, contrastive_modality_reps):
        # shape is [num_users, 1 + n_negs, 2, embedding_dim]
        # print(contrastive_modality_reps.shape)
        first_emb, second_emb = contrastive_modality_reps[..., 0, :], contrastive_modality_reps[..., 1, :]
        logits = first_emb @ second_emb.transpose(-2, -1) / self.cl_temp

        # Positive keys are the entries on the diagonal, therefore these are the correct labels:
        # [batch_size, 0, 1, ..., batch_size - 1]
        labels = torch.arange(logits.shape[-1], device=first_emb.device).repeat(1, *logits.shape[:-2], 1)

        # Logits change depending on which modality we are "retrieving"
        logits_c_to_p = logits.reshape(-1, logits.shape[-1])
        logits_p_to_c = logits.transpose(-2, -1).reshape(-1, logits.shape[-1])

        labels = labels.reshape(-1)
        x_y_loss = F.cross_entropy(logits_c_to_p, labels)
        y_x_loss = F.cross_entropy(logits_p_to_c, labels)

        contrastive_loss = x_y_loss + y_x_loss
        return contrastive_loss

    def predict(self, gu, gi, **kwargs):
        # return torch.sigmoid(torch.matmul(gu.to(self.device), torch.transpose(gi.to(self.device), 0, 1)))
        return torch.matmul(gu.to(self.device), torch.transpose(gi.to(self.device), 0, 1))

    def train_step(self, batch):
        user, pos, neg = batch

        # 1. Forward Pass (Ricalcolo degli embedding con i pesi attuali)
        user_repr = self.get_user_representations(user)
        pos_item_repres_full = self.get_item_representations(pos)
        neg_item_repres_full = self.get_item_representations(neg)

        # 2. BPR always uses the full aggregation over all modalities, matching inference
        pos_item_repr = self._aggregate(pos_item_repres_full, dim=-2)
        neg_item_repr = self._aggregate(neg_item_repres_full, dim=-2)

        # 3. Contrastive pair selection
        interactions_idx = len(self.modalities)
        if self.cl_always_interactions_vs_agg:
            other_idx = list(range(len(self.modalities)))
            pos_interactions = pos_item_repres_full[..., interactions_idx, :]
            pos_others_agg = self._aggregate(pos_item_repres_full[..., other_idx, :], dim=-2)
            neg_interactions = neg_item_repres_full[..., interactions_idx, :]
            neg_others_agg = self._aggregate(neg_item_repres_full[..., other_idx, :], dim=-2)
            pos_item_repres = torch.stack((pos_interactions, pos_others_agg), dim=-2)
            neg_item_repres = torch.stack((neg_interactions, neg_others_agg), dim=-2)
        elif self.cl_force_interactions_modality:
            other_idx = np.random.choice(len(self.modalities), 1, replace=False)[0]
            sampled_modalities_ids = np.array([interactions_idx, other_idx])
            pos_item_repres = pos_item_repres_full[..., sampled_modalities_ids, :]
            neg_item_repres = neg_item_repres_full[..., sampled_modalities_ids, :]
        else:
            sampled_modalities_ids = np.random.choice(len(self.modalities) + 1, 2, replace=False)
            pos_item_repres = pos_item_repres_full[..., sampled_modalities_ids, :]
            neg_item_repres = neg_item_repres_full[..., sampled_modalities_ids, :]
        '''
        xu_pos_score = torch.sum(user_repr * pos_item_repr, 1) # diventare [128,5] con rep
        xu_neg = torch.sum(user_repr * neg_item_repr, 1) # dovrebbe essere [128,5]
        
        xu_pos = xu_pos_score.unsqueeze(1).expand(-1, xu_neg.shape[1])     # NEW
        diff_logits = xu_pos - xu_neg                    # consequently [128,5]
        '''

        # USER FIX
        user_repr = user_repr.squeeze(1)       

        # POS SCORE
        xu_pos_score = torch.sum(user_repr * pos_item_repr, dim=-1)  
        xu_pos_single = xu_pos_score.squeeze()                        
        xu_pos = xu_pos_single.unsqueeze(1).expand(-1, neg_item_repr.shape[1]) 
     
        # NEG SCORE
        user_exp = user_repr.unsqueeze(1)       # [128,1,128]
        xu_neg = torch.sum(user_exp * neg_item_repr, dim=2)            # [128,5]

        # DIFF
        diff_logits = xu_pos - xu_neg           # [128,5]


        # 5. Calcolo Loss
        pos_labels = torch.ones(xu_pos.shape, device=xu_pos.device)


        loss = nn.BCEWithLogitsLoss(reduction='mean')(diff_logits.flatten(), pos_labels.flatten())
        # Stampa delle dimensioni (shape) e del valore della loss BCE
        # print(f"[DEBUG LOSS - BCE] Shape: {loss.shape}, Valore: {loss.item():.4f}")


        contrastive_modality_reps = torch.cat((pos_item_repres[:, None, :, :], neg_item_repres), 1) 
        contrastive_loss = self.loss_contrastive(contrastive_modality_reps)
        # Stampa del valore della loss contrastiva (se reduction='mean')
        # print(f"[DEBUG LOSS - Contrastive] Valore: {contrastive_loss.item():.4f}")

        
        loss += self.cl_weight * contrastive_loss

        # Stampa della loss totale (che sarà anch'essa uno scalare)
        # print(f"[DEBUG LOSS - Totale] Shape: {loss.shape}, Valore: {loss.item():.4f}")

        # 6. Backpropagation (Qui i pesi cambiano!)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().numpy()

    def get_top_k(self, preds, train_mask, k=100):
        return torch.topk(torch.where(torch.tensor(train_mask).to(self.device), preds.to(self.device),
                                      torch.tensor(-np.inf).to(self.device)), k=k, sorted=True)
