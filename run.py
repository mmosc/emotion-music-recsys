# keep in mind: https://github.com/huggingface/lerobot/issues/2762
# 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
from elliot.run import run_experiment
import argparse
import os
from utils import merge_dicts, save_yaml, load_yaml

parser = argparse.ArgumentParser(description="Run training and evaluation.")
parser.add_argument('--config', type=str, default='config_files/tuned/emma_emotion')
parser.add_argument('--seed', type=int, default=123, help='Random seed for reproducibility')
parser.add_argument('--start_fold', type=int, default=None, help='First fold to run (inclusive, 0-indexed)')
parser.add_argument('--end_fold', type=int, default=None, help='Last fold to run (exclusive)')
parser.add_argument('--gpu', type=int, default=None, help='GPU index to use (overrides config)')
parser.add_argument('--warm_restart', action='store_true',
                    help='Resume interrupted training from the last validation-epoch '
                         'resume checkpoint (preserves early-stopping history)')
args = parser.parse_args()

print(f"Running experiment with config: {args.config}.yml")

base_config = load_yaml('config_files/DEFAULT.yml')
extended_config = load_yaml(args.config + '.yml')
config = merge_dicts(base_config, extended_config)

model_defaults_config = load_yaml("config_files/models/model_defaults.yml")

model_keys = config["experiment"]["models"].keys()
for key in model_keys:
    config["experiment"]["models"][key] = {**model_defaults_config, **config["experiment"]["models"][key]}

    if args.seed is not None:
        config["experiment"]["models"][key]['seed'] = args.seed

    if 'modalities' in config.keys():
        config["experiment"]["models"][key]["modalities"] = config['modalities']
    if 'loaders' in config.keys():
        config["experiment"]["models"][key]["loaders"] = config['loaders']

    if args.warm_restart:
        config["experiment"]["models"][key].setdefault("meta", {})["warm_restart"] = True

if args.gpu is not None:
    config["experiment"]["gpu"] = args.gpu

# Use a per-GPU-per-fold temp file to avoid conflicts when the same GPU appears
# multiple times in a parallel run (e.g. --gpus 0 0 1 1 2 2 3 3)
config_tag = args.config.replace("/", "_").replace("config_files_", "")
suffix = f"_{config_tag}_gpu{args.gpu}_fold{args.start_fold}" if args.gpu is not None else f"_{config_tag}"
tmp_config_path = f"config_files/tmp{suffix}.yml"
save_yaml(config, tmp_config_path)

try:
    run_experiment(tmp_config_path, start_fold=args.start_fold, end_fold=args.end_fold, run_tag=config_tag)
finally:
    if os.path.exists(tmp_config_path):
        os.remove(tmp_config_path)