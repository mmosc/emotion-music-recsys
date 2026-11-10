"""Precompute and save the k-core-filtered 10-fold splits of a dataset.

Usage:
    python precompute_splits.py --datasets onion emma
"""

import argparse
import importlib
from os import path

from utils import load_yaml, save_yaml

SPLITS_FOLDER = "/opt/datasets/Music4All/music4all/onion_elliot/{0}/splits"


def build_config(dataset: str) -> dict:
    config = load_yaml("config_files/DEFAULT.yml")
    config["experiment"]["dataset"] = dataset
    config["experiment"]["splitting"]["save_on_disk"] = True
    config["experiment"]["splitting"]["save_folder"] = SPLITS_FOLDER
    return config


def precompute(dataset: str):
    print(f"\n{'=' * 72}\nPrecomputing split for dataset: {dataset}\n{'=' * 72}")

    config = build_config(dataset)
    tmp_path = f"config_files/tmp_precompute_split_{dataset}.yml"
    save_yaml(config, tmp_path)

    import elliot.run as _em
    elliot_here = path.abspath(path.dirname(_em.__file__))

    from elliot.namespace.namespace_model_builder import NameSpaceBuilder

    builder = NameSpaceBuilder(tmp_path, elliot_here, path.abspath(path.dirname(tmp_path)))
    base_ns = builder.base.base_namespace

    dataloader_class = getattr(importlib.import_module("elliot.dataset"), base_ns.data_config.dataloader)
    dataloader_class(config=base_ns).generate_dataobjects()

    print(f"Saved split -> {base_ns.splitting.save_folder}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["onion", "emma"],
                        help="Datasets to precompute splits for (default: onion emma)")
    args = parser.parse_args()

    for dataset in args.datasets:
        precompute(dataset)


if __name__ == "__main__":
    main()
