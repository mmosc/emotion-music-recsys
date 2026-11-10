"""Offline full-metric evaluation of saved recommendation files.

Usage:
    python run_eval.py --tuned_config config_files/tuned/full/sibrar_onion_e \\
        --fold 0 --path /opt/scratch/marta/results/onion/full_e/recs/<recs>_fold=0.tsv
"""
from elliot.run import run_experiment
import argparse
import os
from utils import merge_dicts, save_yaml, load_yaml

_DATASET_EMOTION_DIR = {
    "onion": "gems",
    "emma": "emma",
    "emma_gems": "gems",
    "onion186": "gems",
}

# Complex metrics whose feature_data must track the training emotion space.
_CALIBRATION_METRICS = ("Calibration", "EnergyCalibration")


def resolve_calibration_feature_data(config: dict, default: str) -> str:
    """The emotion features the calibration metrics should score against."""
    experiment = config.get("experiment", {})

    side_information = experiment.get("data_config", {}).get("side_information") or []
    for entry in side_information:
        if isinstance(entry, dict) and entry.get("emotion_features"):
            return entry["emotion_features"]

    annotation_dir = _DATASET_EMOTION_DIR.get(experiment.get("dataset"))
    if annotation_dir is None:
        return default
    return default.rsplit("/", 1)[0] + "/" + annotation_dir


def apply_calibration_feature_data(evaluation: dict, config: dict) -> dict:
    """Point every calibration metric in `evaluation` at `config`'s emotion space."""
    for metric in evaluation.get("complex_metrics", []):
        if metric.get("metric") in _CALIBRATION_METRICS and "feature_data" in metric:
            metric["feature_data"] = resolve_calibration_feature_data(
                config, metric["feature_data"]
            )
    return evaluation


def build_parser():
    parser = argparse.ArgumentParser(description="Offline evaluation of saved recommendations.")
    parser.add_argument('--tuned_config', type=str, required=True,
                        help='Original training config path without .yml '
                             '(e.g. config_files/tuned/full/sibrar_onion_e)')
    parser.add_argument('--fold', type=int, required=True,
                        help='Test fold index the recs file was generated on')
    parser.add_argument('--path', type=str, required=True,
                        help='Path to the saved recommendations .tsv')
    parser.add_argument('--eval_config', type=str, default='config_files/eval',
                        help='Config supplying the full evaluation metric suite (without .yml)')
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    config = merge_dicts(load_yaml('config_files/DEFAULT.yml'), load_yaml(args.tuned_config + '.yml'))
    eval_config = load_yaml(args.eval_config + '.yml')

    evaluation = eval_config['experiment']['evaluation']
    config['experiment']['evaluation'] = apply_calibration_feature_data(evaluation, config)
    config['experiment']['models'] = {'ProxyRecommender': {'path': args.path}}

    config_tag = args.tuned_config.replace("/", "_").replace("config_files_", "")
    run_tag = f"eval_{config_tag}"
    tmp_config_path = f"config_files/tmp_{run_tag}_fold{args.fold}.yml"
    save_yaml(config, tmp_config_path)

    try:
        run_experiment(tmp_config_path, start_fold=args.fold, end_fold=args.fold + 1, run_tag=run_tag)
    finally:
        if os.path.exists(tmp_config_path):
            os.remove(tmp_config_path)


if __name__ == '__main__':
    main()
