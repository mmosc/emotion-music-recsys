"""Apply dimensionality reduction to Elliot modality feature folders.

Usage
-----
    python reduce_modality_features.py --modality audio \\
        --input-dir /opt/datasets/Music4All/music4all/onion_elliot/0/audio_embeddings/maest \\
        --n-components 64

    # custom output location
    python reduce_modality_features.py --modality visual \\
        --input-dir .../visual_embeddings/resnet \\
        --output-dir .../visual_embeddings/resnet_pca32 \\
        --n-components 32
"""

import argparse
import os

import numpy as np

from dimensionality_reduction import REDUCERS


def _load_modality_features(input_dir: str):
    """Load every `{item_id}.npy` file in `input_dir` into one matrix."""
    filenames = sorted(f for f in os.listdir(input_dir) if f.endswith(".npy"))
    if not filenames:
        raise ValueError(f"No .npy files found in {input_dir}")

    item_ids = [int(f.split(".")[0]) for f in filenames]
    vectors = [np.load(os.path.join(input_dir, f), allow_pickle=True) for f in filenames]
    features = np.stack([v.reshape(-1) for v in vectors])
    return item_ids, features


def _save_modality_features(output_dir: str, item_ids, features: np.ndarray):
    os.makedirs(output_dir, exist_ok=True)
    for item_id, vector in zip(item_ids, features):
        np.save(os.path.join(output_dir, f"{item_id}.npy"), vector)


def reduce_modality(modality: str, input_dir: str, output_dir: str,
                     method: str, n_components: int) -> str:
    print(f"[{modality}] loading features from {input_dir}")
    item_ids, features = _load_modality_features(input_dir)
    print(f"[{modality}] loaded {features.shape[0]} items, dim={features.shape[1]}")

    reducer_cls = REDUCERS[method]
    reducer = reducer_cls(n_components=n_components)
    reduced = reducer.fit_transform(features)

    output_dir = output_dir or f"{input_dir.rstrip('/')}_{reducer.name}{n_components}"
    _save_modality_features(output_dir, item_ids, reduced)
    print(f"[{modality}] saved {reduced.shape[0]} items, dim={reduced.shape[1]} -> {output_dir}")
    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Reduce dimensionality of an Elliot modality feature folder.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--modality", choices=["audio", "textual", "visual"], required=True,
        help="Modality being processed (used only for logging/output naming).",
    )
    parser.add_argument(
        "--input-dir", required=True,
        help="Folder of per-item {item_id}.npy feature files to reduce.",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Folder to write reduced {item_id}.npy files to. "
             "Defaults to '<input-dir>_<method><n-components>'.",
    )
    parser.add_argument(
        "--method", choices=list(REDUCERS.keys()), default="pca",
        help="Dimensionality reduction technique to apply (default: pca).",
    )
    parser.add_argument(
        "--n-components", type=int, required=True,
        help="Number of output dimensions.",
    )
    args = parser.parse_args()

    reduce_modality(
        modality=args.modality,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        method=args.method,
        n_components=args.n_components,
    )


if __name__ == "__main__":
    main()
