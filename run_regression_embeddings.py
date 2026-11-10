"""Item-level emotion regression from CF item embeddings.

Usage:
    python run_regression_embeddings.py                 # all 10 recommender folds
    python run_regression_embeddings.py --folds 0       # one fold (fast smoke run)
"""

import argparse
import json
import os

import numpy as np

from emotion_regression import (
    N_ENTITY_FOLDS,
    entity_folds,
    fit_mlp,
    paired_effect_size,
    paired_wilcoxon,
    run_regression_grid,
)
from emotion_regression_data import DEFAULT_DATASET, load_item_embeddings, scratch_dir

BASELINES = ("train_mean", "gaussian_independent", "gaussian_correlated", "random_projection")



def describe_group_balance(gems, groups):
    """Per-entity-fold target means (checks that the ID-modulo split is balanced)."""
    spreads = []
    for dim in range(gems.shape[1]):
        per_group = [gems[groups == k][:, dim].mean() for k in range(N_ENTITY_FOLDS)]
        spreads.append(float(np.std(per_group)))
    return {
        "per_dim_group_mean_spread": spreads,
        "per_dim_overall_std": [float(s) for s in gems.std(axis=0)],
    }


def run(recommender_folds, model="LightGCN", seed=0, regressor="linear",
        dataset=DEFAULT_DATASET):
    """regressor="linear": least squares; regressor="mlp": MLP without hyperparameter search."""
    fit_fn = None if regressor == "linear" else (lambda xt, yt: fit_mlp(xt, yt, seed=seed))
    grid = {}
    balance = None
    for fold in recommender_folds:
        emb, gems, item_ids = load_item_embeddings(model, fold, dataset=dataset)
        if balance is None:
            balance = describe_group_balance(gems, entity_folds(item_ids))
        result = run_regression_grid(emb, gems, item_ids, seed=seed, fit_fn=fit_fn)
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
        regressor_mean = np.nanmean(result["regressor"]["overall"])
        train_mean_fit = np.nanmean(result["regressor"]["overall_train"])
        print(f"[fold {fold}] regressor R2 = {regressor_mean:+.4f} (train-fit R2 = {train_mean_fit:+.4f})", flush=True)
        for name in BASELINES:
            print(f"           {name:>22} = {np.nanmean(result[name]['overall']):+.4f}", flush=True)

    return {"grid": grid, "group_balance": balance, "model": model,
            "seed": seed, "regressor": regressor, "dataset": dataset,
            "n_entities": int(len(item_ids))}


def summarize(payload):
    """Pool the 10x10 cells per predictor and pair the regressor with each baseline."""
    grid = payload["grid"]
    pooled = {name: [] for name in ("regressor",) + BASELINES}
    for fold_result in grid.values():
        for name in pooled:
            pooled[name].extend(fold_result[name]["overall"])

    lines = []
    reg = np.asarray(pooled["regressor"], dtype=np.float64)
    n_ent = payload.get("n_entities")
    if n_ent:
        per_group = n_ent / 10.0
        lines.append(f"n = {n_ent} entities, ~{per_group:.0f} per test group "
                     f"(10 recommender folds x 10 entity groups = {reg.size} cells)")
        lines.append("cells are NOT independent: they resample the same entities.")
        lines.append("")
    lines.append(f"regressor               mean={np.nanmean(reg):+.4f} std={np.nanstd(reg):.4f} n={reg.size}")
    for name in BASELINES:
        vals = np.asarray(pooled[name], dtype=np.float64)
        ok = ~(np.isnan(reg) | np.isnan(vals))
        stat, p = paired_wilcoxon(reg[ok], vals[ok])
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
    parser.add_argument("--folds", type=int, nargs="*", default=list(range(10)),
                        help="recommender folds supplying the checkpoints")
    parser.add_argument("--model", default="LightGCN")
    parser.add_argument("--regressor", choices=("linear", "mlp"), default="linear",
                        help="linear = least squares; mlp = Task 3 capacity check")
    parser.add_argument("--dataset", default=DEFAULT_DATASET,
                        help="Dataset (default: %(default)s)")
    parser.add_argument("--out", default=None,
                        help="where to write the raw grid (default: under the dataset's regression output dir)")
    args = parser.parse_args()

    out_path = args.out or os.path.join(
        scratch_dir(args.dataset), f"regression_{args.model.lower()}_{args.regressor}.json"
    )
    payload = run(args.folds, model=args.model, regressor=args.regressor, dataset=args.dataset)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(payload, fh, indent=2)

    print("\n" + summarize(payload))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
