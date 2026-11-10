"""Data loaders for the emotion regression experiments."""

import glob
import importlib
import os

import numpy as np
import scipy.sparse as sp
import torch
import yaml

from utils import RESULTS_BASE, load_yaml, merge_dicts

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

DEFAULT_DATASET = "onion"
# Both datasets live under the same `onion_elliot` root; `{0}` is the dataset
# name, exactly as in the training configs.
_SPLITS_ROOT_TEMPLATE = "/opt/datasets/Music4All/music4all/onion_elliot/{0}/splits"
_SUPPORTED_DATASETS = ("onion", "emma", "emma_gems", "onion186")

_BASE_DATASET = {"emma_gems": "emma"}

# Which `emotion_embeddings/` subdirectory supplies each dataset's targets.
_ANNOTATION_SET = {
    "onion": "gems",
    "emma": "emma",
    "emma_gems": "gems",
    "onion186": "gems",
}


def _validate_dataset(dataset):
    if dataset not in _SUPPORTED_DATASETS:
        raise ValueError(
            f"Unsupported dataset {dataset!r}; supported: {_SUPPORTED_DATASETS}. "
            "Refusing to fall back to a default -- silently probing the wrong "
            "dataset's annotations produces a plausible but meaningless result."
        )
    return dataset


def base_dataset(dataset=DEFAULT_DATASET):
    """The dataset whose splits and checkpoints `dataset` reads."""
    _validate_dataset(dataset)
    return _BASE_DATASET.get(dataset, dataset)


def annotation_set(dataset=DEFAULT_DATASET):
    """The `emotion_embeddings/` subdirectory supplying this dataset's targets."""
    _validate_dataset(dataset)
    return _ANNOTATION_SET[dataset]


def emotion_config_path(dataset=DEFAULT_DATASET):
    """The dataset config registering EmotionAttribute for this dataset."""
    _validate_dataset(dataset)
    return os.path.join(_REPO_ROOT, "config_files", "datasets", f"{dataset}_emotion.yml")


def splits_root(dataset=DEFAULT_DATASET):
    # Input: resolved through base_dataset, so emma_gems reads EMMA's folds.
    return _SPLITS_ROOT_TEMPLATE.format(base_dataset(dataset))


def weights_root(dataset=DEFAULT_DATASET):
    # emma_gems reuses EMMA's checkpoints.
    return os.path.join(RESULTS_BASE, base_dataset(dataset), "full", "weights")


def scratch_dir(dataset=DEFAULT_DATASET):
    """Output directory of the emotion regression runs for `dataset`."""
    _validate_dataset(dataset)
    return os.path.join(RESULTS_BASE, dataset, "emotion_regression")



_SUPPORTED_MODELS = ("LightGCN", "DeepMF")


_DATASET_CACHE = {}


def _cache_key(dataset):
    _validate_dataset(dataset)
    return (dataset, "all")


def _build_all_fold_datasets(dataset=DEFAULT_DATASET):
    """Build every fold's Elliot DataSet with EmotionAttribute registered."""
    import elliot
    from elliot.namespace.namespace_model_builder import NameSpaceBuilder

    _validate_dataset(dataset)
    base_config = load_yaml(os.path.join(_REPO_ROOT, "config_files", "DEFAULT.yml"))
    ext_config = load_yaml(emotion_config_path(dataset))
    config = merge_dicts(base_config, ext_config)
    config["experiment"]["data_config"]["strategy"] = "hierarchy"
    config["experiment"]["data_config"]["root_folder"] = splits_root(dataset)

    # Per-dataset temp file: concurrent regression runs on the two datasets would
    # otherwise race on one path and build the wrong dataset's folds.
    tmp_path = os.path.join(_REPO_ROOT, "config_files", f"tmp_emotion_regression_{dataset}.yml")
    with open(tmp_path, "w") as fh:
        yaml.safe_dump(config, fh)
    elliot_dir = os.path.dirname(elliot.__file__)
    try:
        builder = NameSpaceBuilder(tmp_path, elliot_dir, os.path.abspath(os.path.join(_REPO_ROOT, "config_files")))
        base = builder.base
        dataloader_class = getattr(
            importlib.import_module("elliot.dataset"), base.base_namespace.data_config.dataloader
        )
        dataloader = dataloader_class(config=base.base_namespace)
        data_test_list = dataloader.generate_dataobjects()
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return data_test_list


def _build_fold_dataset(fold, dataset=DEFAULT_DATASET):
    """Return one fold's DataSet, building (and caching) all folds on first call."""
    key = _cache_key(dataset)
    if key not in _DATASET_CACHE:
        _DATASET_CACHE[key] = _build_all_fold_datasets(dataset)
    data_test_list = _DATASET_CACHE[key]
    if not 0 <= fold < len(data_test_list):
        raise IndexError(f"fold {fold} out of range: {dataset} has {len(data_test_list)} folds")
    return data_test_list[fold][0]


def _find_checkpoint(model, fold, dataset=DEFAULT_DATASET):
    root = weights_root(dataset)
    pattern = os.path.join(
        root, f"{model}_*_fold={fold}", f"best-weights-{model}_*_fold={fold}"
    )
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"No {model} checkpoint found for fold {fold} under {root} "
            f"(pattern: {pattern})"
        )
    if len(matches) > 1:
        raise FileNotFoundError(
            f"Multiple {model} checkpoints matched fold {fold}, expected exactly one: {matches}"
        )
    return matches[0]


def _find_lightgcn_checkpoint(fold, dataset=DEFAULT_DATASET):
    return _find_checkpoint("LightGCN", fold, dataset)


def _load_lightgcn_embeddings(fold, dataset=DEFAULT_DATASET):
    """Return (user_emb, item_emb), row-aligned to the DataSet's users/items."""
    checkpoint_path = _find_lightgcn_checkpoint(fold, dataset)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "model_state_dict" not in checkpoint:
        raise KeyError(f"Checkpoint {checkpoint_path} has no 'model_state_dict' key")
    state_dict = checkpoint["model_state_dict"]
    if "Gu.weight" not in state_dict or "Gi.weight" not in state_dict:
        raise KeyError(f"Checkpoint {checkpoint_path} is missing 'Gu.weight'/'Gi.weight'")
    user_emb = state_dict["Gu.weight"].numpy()
    item_emb = state_dict["Gi.weight"].numpy()
    return user_emb, item_emb


def _load_deepmf_embeddings(fold, urm=None, dataset=DEFAULT_DATASET):
    """Return (user_emb, item_emb) for DeepMF: the *output* of its trained MLPs."""
    checkpoint_path = _find_checkpoint("DeepMF", fold, dataset)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "model_state_dict" not in checkpoint:
        raise KeyError(f"Checkpoint {checkpoint_path} has no 'model_state_dict' key")
    state_dict = checkpoint["model_state_dict"]

    def _mlp_weights(prefix):
        keys = [k for k in state_dict if k.startswith(f"{prefix}.") and k.endswith(".weight")]
        if not keys:
            raise KeyError(
                f"Checkpoint {checkpoint_path} has no '{prefix}' layers; DeepMF's structure "
                f"differs from the expected Linear/ReLU/Linear stack."
            )
        layers = []
        for key in sorted(keys, key=lambda k: int(k.split(".")[1])):
            idx = key.split(".")[1]
            layers.append((
                state_dict[key].clone(),
                state_dict[f"{prefix}.{idx}.bias"].clone(),
            ))
        return layers

    user_layers = _mlp_weights("user_embedding_module")
    item_layers = _mlp_weights("item_embedding_module")
    # Free the frozen URM copies (the bulk of the 4.5 GB) before the forward pass.
    del state_dict, checkpoint

    if urm is None:
        urm = _build_fold_dataset(fold, dataset).sp_i_train

    user_emb = _apply_mlp(urm, user_layers)
    item_emb = _apply_mlp(urm.T.tocsr(), item_layers)
    return user_emb, item_emb


def _apply_mlp(sparse_rows, layers, batch_size=4096):
    """Run a sparse row-block through a Linear/ReLU/.../Linear stack."""
    sparse_rows = sparse_rows.tocsr()
    n_rows = sparse_rows.shape[0]
    out_dim = layers[-1][0].shape[0]
    out = np.empty((n_rows, out_dim), dtype=np.float32)

    with torch.no_grad():
        for start in range(0, n_rows, batch_size):
            block = sparse_rows[start:start + batch_size].toarray()
            h = torch.from_numpy(block.astype(np.float32))
            for i, (weight, bias) in enumerate(layers):
                h = torch.nn.functional.linear(h, weight, bias)
                if i < len(layers) - 1:
                    h = torch.relu(h)
            out[start:start + block.shape[0]] = h.numpy()
    return out


def _validate_model(model):
    if model not in _SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model {model!r}; supported: {_SUPPORTED_MODELS}")


def load_item_embeddings(model, fold, dataset=DEFAULT_DATASET):
    """Return (emb, gems, item_ids): emb[i]/gems[i]/item_ids[i] refer to the same item."""
    _validate_model(model)
    data = _build_fold_dataset(fold, dataset)

    if model == "DeepMF":
        _, item_emb_full = _load_deepmf_embeddings(fold, urm=data.sp_i_train, dataset=dataset)
    else:
        _, item_emb_full = _load_lightgcn_embeddings(fold, dataset)

    if item_emb_full.shape[0] != len(data.items):
        raise ValueError(
            f"Checkpoint item-embedding row count ({item_emb_full.shape[0]}) does not match "
            f"fold {fold}'s DataSet item count ({len(data.items)}) — likely an ID misalignment "
            f"between the checkpoint and the rebuilt DataSet."
        )

    gems_full = data.side_information.EmotionAttribute.object.get_all_features(item_order=data.items)

    item_ids = np.asarray(data.items)
    emb = item_emb_full[[data.public_items[i] for i in item_ids]]
    gems = gems_full[[data.public_items[i] for i in item_ids]]

    return emb, gems, item_ids


def load_urm_column_index(fold, dataset=DEFAULT_DATASET):
    """Return {raw_item_id: URM column}, so callers can align track IDs to URM columns."""
    data = _build_fold_dataset(fold, dataset)
    return dict(data.public_items)


def load_urm(fold, dataset=DEFAULT_DATASET):
    """Return the fold's training URM, column-aligned with load_item_embeddings."""
    data = _build_fold_dataset(fold, dataset)
    urm = data.sp_i_train
    if not sp.issparse(urm):
        raise TypeError(f"Expected data.sp_i_train to be scipy.sparse, got {type(urm)}")
    return urm
