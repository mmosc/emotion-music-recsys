"""Emotion regression from raw streams: each track's column R_{*t} of the user-track matrix.

Usage:
    python run_regression_streams.py --dataset emma
"""

import argparse
import json
import os

import numpy as np
import scipy.sparse as sp

from emotion_regression import fit_mlp, paired_effect_size, paired_wilcoxon, run_regression_grid
from emotion_regression_data import (
    DEFAULT_DATASET,
    load_item_embeddings,
    load_urm,
    load_urm_column_index,
    scratch_dir,
)

BASELINES = ("train_mean", "gaussian_independent", "gaussian_correlated", "random_projection")


def build_track_input(fold, model="LightGCN", dataset=DEFAULT_DATASET):
    """x = each track's URM column (who streamed it), y = that track's GEMS vector."""
    _, gems, item_ids = load_item_embeddings(model, fold, dataset=dataset)
    urm = load_urm(fold, dataset=dataset)
    col_index = load_urm_column_index(fold, dataset=dataset)

    # Transpose so rows are items; stays sparse throughout.
    urm_by_item = urm.T.tocsr()
    rows = np.asarray([col_index[i] for i in item_ids])
    x = urm_by_item[rows]
    assert sp.issparse(x), "raw streams input must stay sparse"
    return x, gems, item_ids


def run(recommender_folds, model="LightGCN", seed=0,
        dataset=DEFAULT_DATASET, regressor="linear"):
    """regressor="linear": least squares; regressor="mlp": MLP without hyperparameter search."""
    fit_fn = None if regressor == "linear" else (lambda xt, yt: fit_mlp(xt, yt, seed=seed))
    grid = {}
    for fold in recommender_folds:
        x, y, entity_ids = build_track_input(fold, model=model, dataset=dataset)
        print(f"[fold {fold}] input: x={x.shape} sparse={sp.issparse(x)} "
              f"nnz={x.nnz} y={y.shape}", flush=True)
        result = run_regression_grid(x, y, entity_ids, seed=seed, sparse_solver=True,
                                fit_fn=fit_fn)
        grid[fold] = {
            name: {
                "overall": result[name]["overall"].tolist(),
                "per_dim": result[name]["per_dim"].tolist(),
                "overall_train": result[name]["overall_train"].tolist(),
                "overall_val": result[name]["overall_val"].tolist(),
                "alpha": [None if a is None else str(a) for a in result[name]["alpha"]],
            }
            for name in result
        }
        print(f"[fold {fold}] regressor R2 = {np.nanmean(result['regressor']['overall']):+.4f}", flush=True)
        for name in BASELINES:
            print(f"           {name:>22} = {np.nanmean(result[name]['overall']):+.4f}", flush=True)

    return {"grid": grid, "model": model, "seed": seed,
            "dataset": dataset, "regressor": regressor, "n_entities": int(len(entity_ids))}


def summarize(payload):
    grid = payload["grid"]
    pooled = {name: [] for name in ("regressor",) + BASELINES}
    for fold_result in grid.values():
        for name in pooled:
            pooled[name].extend(fold_result[name]["overall"])

    reg = np.asarray(pooled["regressor"], dtype=np.float64)
    lines = []
    n_ent = payload.get("n_entities")
    if n_ent:
        lines.append(f"n = {n_ent} entities, ~{n_ent/10:.0f} per test group; "
                     f"{reg.size} cells, NOT independent (they resample the same entities).")
    lines.append("")
    lines.append(f"regressor               mean={np.nanmean(reg):+.4f} std={np.nanstd(reg):.4f} n={reg.size}")
    for name in BASELINES:
        vals = np.asarray(pooled[name], dtype=np.float64)
        ok = ~(np.isnan(reg) | np.isnan(vals))
        _, p = paired_wilcoxon(reg[ok], vals[ok])
        es = paired_effect_size(reg[ok], vals[ok])
        lines.append(
            f"{name:<22}  mean={np.nanmean(vals):+.4f} std={np.nanstd(vals):.4f} "
            f"| delta={es['median_diff']:+.4f} "
            f"[95% CI {es['ci_low']:+.4f}, {es['ci_high']:+.4f}] "
            f"cliff={es['cliffs_delta']:+.3f} p={p:.3g}"
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, nargs="*", default=list(range(10)))
    parser.add_argument("--model", default="LightGCN")
    parser.add_argument("--dataset", default=DEFAULT_DATASET,
                        help="Dataset (default: %(default)s)")
    parser.add_argument("--regressor", choices=("linear", "mlp"), default="linear",
                        help="linear = least squares; mlp = MLP without hyperparameter search")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    suffix = "" if args.regressor == "linear" else f"_{args.regressor}"
    out_path = args.out or os.path.join(
        scratch_dir(args.dataset), f"regression_streams{suffix}.json"
    )
    payload = run(args.folds, model=args.model, dataset=args.dataset,
                  regressor=args.regressor)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(payload, fh, indent=2)

    print("\n" + summarize(payload))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
