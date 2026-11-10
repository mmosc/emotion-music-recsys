"""Item-level emotion regression with a grid-searched MLP.

Usage:
    python run_regression_mlp.py --dataset emma --model LightGCN
    python run_regression_mlp.py --dataset onion186 --model RawURM
"""

import argparse
import json
import os
import time

import numpy as np

from emotion_regression import (
    MLP_HPO_GRID,
    N_ENTITY_FOLDS,
    entity_folds,
    fit_mlp_with,
    run_regression_grid,
    tune_mlp,
)
from emotion_regression_data import DEFAULT_DATASET, load_item_embeddings, scratch_dir
from run_regression_streams import build_track_input

BASELINES = ("train_mean", "gaussian_independent", "gaussian_correlated", "random_projection")


def load_arm(model, fold, dataset):
    """(x, y, entity_ids) for one CF regressor at one recommendation fold."""
    if model == "RawURM":
        return build_track_input(fold, dataset=dataset)
    emb, gems, ids = load_item_embeddings(model, fold, dataset=dataset)
    return emb, gems, ids


def tune_per_entity_fold(model, dataset, seed=0, grid=None, n_splits=5, n_repeats=3):
    """One configuration per entity fold, chosen on recommendation fold 0."""
    x, y, ids = load_arm(model, 0, dataset)
    groups = entity_folds(ids)
    chosen = {}
    for k in range(N_ENTITY_FOLDS):
        train = groups != k
        x_tr = x[np.flatnonzero(train)] if hasattr(x, "tocsr") else x[train]
        t0 = time.time()
        params, score = tune_mlp(x_tr, np.asarray(y)[train], seed=seed, grid=grid,
                                 n_splits=n_splits, n_repeats=n_repeats)
        chosen[k] = {"params": {p: str(v) for p, v in params.items()},
                     "_raw": params, "inner_cv_r2": score}
        print(f"  [entity fold {k}] inner-CV R2 = {score:+.4f}  "
              f"hidden={params['mlpregressor__hidden_layer_sizes']} "
              f"alpha={params['mlpregressor__alpha']} "
              f"max_iter={params['mlpregressor__max_iter']}  ({time.time()-t0:.0f}s)",
              flush=True)
    return chosen


def run(model, dataset, chosen, seed=0, folds=range(10)):
    grid_out = {}
    for fold in folds:
        x, y, ids = load_arm(model, fold, dataset)

        # Declaring `fold` makes run_regression_grid pass the current entity fold, so
        # each fold is fitted with the configuration selected for it.
        def fit_fn(x_tr, y_tr, fold):
            entry = chosen[fold]
            model_ = fit_mlp_with(entry["_raw"], x_tr, y_tr, seed=seed)
            model_._val_score = entry["inner_cv_r2"]
            return model_, json.dumps(entry["params"], sort_keys=True)

        result = run_regression_grid(x, y, ids, seed=seed, fit_fn=fit_fn,
                                sparse_solver=hasattr(x, "tocsr"))
        grid_out[fold] = {
            name: {
                "overall": result[name]["overall"].tolist(),
                "per_dim": result[name]["per_dim"].tolist(),
                "overall_train": result[name]["overall_train"].tolist(),
                "overall_val": result[name]["overall_val"].tolist(),
                "alpha": [None if a is None else str(a) for a in result[name]["alpha"]],
            }
            for name in result
        }
        print(f"[rec fold {fold}] regressor R2 = {np.nanmean(result['regressor']['overall']):+.4f} "
              f"(train {np.nanmean(result['regressor']['overall_train']):+.4f}, "
              f"val {np.nanmean(result['regressor']['overall_val']):+.4f})", flush=True)
    return grid_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--model", default="LightGCN", choices=("LightGCN", "DeepMF", "RawURM"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--n-repeats", type=int, default=3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    print(f"=== HPO: {args.dataset} / {args.model} / seed {args.seed} ===", flush=True)
    print(f"grid = {MLP_HPO_GRID}", flush=True)

    grid = dict(MLP_HPO_GRID)
    t0 = time.time()
    chosen = tune_per_entity_fold(args.model, args.dataset, seed=args.seed, grid=grid,
                                  n_splits=args.n_splits, n_repeats=args.n_repeats)
    print(f"tuning done in {time.time()-t0:.0f}s", flush=True)

    grid_out = run(args.model, args.dataset, chosen, seed=args.seed)

    out = args.out or os.path.join(
        scratch_dir(args.dataset),
        f"regression_{args.model.lower()}_mlp_hpo_seed{args.seed}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    payload = {
        "grid": grid_out, "model": args.model, "dataset": args.dataset,
        "regressor": "mlp_hpo", "seed": args.seed,
        "hpo_grid": {k: [str(v) for v in vs] for k, vs in grid.items()},
        "selected_per_entity_fold": {
            str(k): {"params": v["params"], "inner_cv_r2": v["inner_cv_r2"]}
            for k, v in chosen.items()},
    }
    with open(out, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
