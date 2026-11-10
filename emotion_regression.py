"""Emotion regression: predict GEMS profiles from streaming data."""

import inspect

import numpy as np
import scipy.sparse as sp
from scipy.stats import wilcoxon
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, RepeatedKFold, cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

N_ENTITY_FOLDS = 10

def entity_folds(entity_ids, n_folds=N_ENTITY_FOLDS):
    """Assign each entity to one of `n_folds` disjoint groups, deterministically."""
    return np.asarray(entity_ids).astype(np.int64) % n_folds


def r2_overall(y_true, y_pred):
    """R^2 pooled across output dimensions (variance-weighted)."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    sse = np.sum((y_true - y_pred) ** 2)
    sst = np.sum((y_true - y_true.mean(axis=0)) ** 2)
    if sst == 0:
        return np.nan
    return 1.0 - sse / sst


def r2_per_dimension(y_true, y_pred):
    """R^2 for each output dimension separately (length = n target dims)."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    sse = np.sum((y_true - y_pred) ** 2, axis=0)
    sst = np.sum((y_true - y_true.mean(axis=0)) ** 2, axis=0)
    out = np.full(y_true.shape[1], np.nan, dtype=np.float64)
    nonzero = sst > 0
    out[nonzero] = 1.0 - sse[nonzero] / sst[nonzero]
    return out


def fit_linear(x_train, y_train, cv=3):
    """Least-squares linear regression (no regularisation)."""
    model = LinearRegression()
    val_score = float(np.mean(cross_val_score(LinearRegression(), x_train, y_train, cv=cv, scoring="r2")))
    model.fit(x_train, y_train)
    model._val_score = val_score
    return model, None


def fit_mlp(x_train, y_train, hidden=(128,), seed=0, max_iter=500):
    """MLP with one hidden layer, early-stopped on a split of the training data."""
    pipeline = make_pipeline(
        StandardScaler(with_mean=not sp.issparse(x_train)),
        MLPRegressor(
            hidden_layer_sizes=hidden,
            random_state=seed,
            max_iter=max_iter,
            early_stopping=True,
            n_iter_no_change=10,
        ),
    )
    pipeline.fit(x_train, y_train)
    pipeline._val_score = float(
        pipeline.named_steps["mlpregressor"].best_validation_score_
    )
    return pipeline, f"hidden={hidden}"


# The MLP's search space: one hidden ReLU layer (sklearn's default activation),
# L2 weight `alpha`, at most 200 training epochs.
MLP_HPO_GRID = {
    "mlpregressor__hidden_layer_sizes": [(8,), (32,), (128,)],
    "mlpregressor__learning_rate_init": [1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
    "mlpregressor__alpha": [1.0, 100.0, 1e-3, 1e-4, 1e-5],
    "mlpregressor__max_iter": [200],
}


def tune_mlp(x_train, y_train, seed=0, grid=None, n_splits=5, n_repeats=3, n_jobs=-1):
    """Select MLP hyperparameters by repeated K-fold CV *inside the training split*."""
    grid = MLP_HPO_GRID if grid is None else grid
    pipeline = make_pipeline(
        StandardScaler(with_mean=not sp.issparse(x_train)),
        MLPRegressor(random_state=seed, early_stopping=False),
    )
    search = GridSearchCV(
        pipeline,
        grid,
        cv=RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed),
        scoring="r2",
        n_jobs=n_jobs,
    )
    search.fit(x_train, y_train)
    return search.best_params_, float(search.best_score_)


def fit_mlp_with(params, x_train, y_train, seed=0):
    """Fit an MLP with a fixed configuration chosen by `tune_mlp`."""
    pipeline = make_pipeline(
        StandardScaler(with_mean=not sp.issparse(x_train)),
        MLPRegressor(random_state=seed, early_stopping=False),
    )
    pipeline.set_params(**params)
    pipeline.fit(x_train, y_train)
    return pipeline


def predict_train_mean(y_train, n_test):
    """Train-mean baseline: predict the training mean for every test entity."""
    mean = np.asarray(y_train, dtype=np.float64).mean(axis=0)
    return np.tile(mean, (n_test, 1))


def predict_gaussian_resample(y_train, n_test, seed, correlated=False):
    """Draw each prediction from a Gaussian fitted on the training split."""
    y_train = np.asarray(y_train, dtype=np.float64)
    rng = np.random.default_rng(seed)
    mean = y_train.mean(axis=0)
    if correlated:
        cov = np.cov(y_train, rowvar=False)
        return rng.multivariate_normal(mean, cov, size=n_test)
    std = y_train.std(axis=0)
    return rng.normal(loc=mean, scale=std, size=(n_test, y_train.shape[1]))


def random_projection_embedding(x_real, seed):
    """Random vectors matching `x_real`'s shape and per-dimension scale."""
    if sp.issparse(x_real):
        rng = np.random.default_rng(seed)
        csr = x_real.tocsr()
        n_rows, n_cols = csr.shape
        row_nnz = np.diff(csr.indptr)
        indices = np.concatenate(
            [rng.choice(n_cols, size=k, replace=False) for k in row_nnz]
        ) if row_nnz.sum() else np.array([], dtype=np.int64)
        data = np.ones(int(row_nnz.sum()), dtype=np.float64)
        return sp.csr_matrix((data, indices, csr.indptr), shape=(n_rows, n_cols))

    x_real = np.asarray(x_real, dtype=np.float64)
    rng = np.random.default_rng(seed)
    std = x_real.std(axis=0)
    return rng.normal(
        loc=x_real.mean(axis=0),
        scale=np.where(std < 1e-12, 1.0, std),
        size=x_real.shape,
    )


def run_regression_grid(x, y, entity_ids, seed=0, sparse_solver=False, fit_fn=None):
    """Run the regression and the baselines over the entity folds."""
    groups = entity_folds(entity_ids)
    n_dims = np.asarray(y).shape[1]
    predictors = ("regressor", "train_mean", "gaussian_independent", "gaussian_correlated", "random_projection")
    out = {
        name: {
            "overall": np.full(N_ENTITY_FOLDS, np.nan),
            "per_dim": np.full((N_ENTITY_FOLDS, n_dims), np.nan),
            "alpha": [None] * N_ENTITY_FOLDS,
            "overall_train": np.full(N_ENTITY_FOLDS, np.nan),
            "overall_val": np.full(N_ENTITY_FOLDS, np.nan),
        }
        for name in predictors
    }

    fit = fit_fn if fit_fn is not None else fit_linear
    _fit_wants_fold = "fold" in inspect.signature(fit).parameters

    def _call_fit(xt, yt, k):
        return fit(xt, yt, fold=k) if _fit_wants_fold else fit(xt, yt)

    for k in range(N_ENTITY_FOLDS):
        train_mask = groups != k
        test_mask = groups == k
        if not train_mask.any() or not test_mask.any():
            continue

        x_train, x_test = _take_rows(x, train_mask), _take_rows(x, test_mask)
        y_train, y_test = np.asarray(y)[train_mask], np.asarray(y)[test_mask]
        n_test = y_test.shape[0]

        model, alpha = _call_fit(x_train, y_train, k)
        _record(out["regressor"], k, y_test, model.predict(x_test), alpha)
        out["regressor"]["overall_train"][k] = r2_overall(y_train, model.predict(x_train))
        out["regressor"]["overall_val"][k] = getattr(model, "_val_score", np.nan)

        _record(out["train_mean"], k, y_test, predict_train_mean(y_train, n_test), None)
        _record(
            out["gaussian_independent"], k, y_test,
            predict_gaussian_resample(y_train, n_test, seed=seed + k, correlated=False), None,
        )
        _record(
            out["gaussian_correlated"], k, y_test,
            predict_gaussian_resample(y_train, n_test, seed=seed + k, correlated=True), None,
        )

        # The random-projection baseline replaces the *input* with uninformative vectors of matching
        # shape and scale, then fits the regressor exactly as usual.
        x_rand = random_projection_embedding(x, seed=seed + k)
        rand_model, rand_alpha = _call_fit(_take_rows(x_rand, train_mask), y_train, k)
        _record(
            out["random_projection"], k, y_test,
            rand_model.predict(_take_rows(x_rand, test_mask)), rand_alpha,
        )

    return out


def _take_rows(x, mask):
    """Row-subset `x`, keeping sparse input sparse (never densified)."""
    if sp.issparse(x):
        return x.tocsr()[np.flatnonzero(mask)]
    return np.asarray(x)[mask]


def _record(entry, k, y_test, y_pred, alpha):
    entry["overall"][k] = r2_overall(y_test, y_pred)
    entry["per_dim"][k] = r2_per_dimension(y_test, y_pred)
    entry["alpha"][k] = alpha


SHARED_RANDOM_DIM = 64


def shared_random_reference(y, entity_ids, seed=0, dim=SHARED_RANDOM_DIM):
    """R^2 of a fixed random input per entity fold, shared across inputs."""
    y = np.asarray(y, dtype=np.float64)
    groups = entity_folds(entity_ids)
    out = np.full(N_ENTITY_FOLDS, np.nan)
    for k in range(N_ENTITY_FOLDS):
        train_mask, test_mask = groups != k, groups == k
        if not train_mask.any() or not test_mask.any():
            continue
        rng = np.random.default_rng(seed + k)
        x_shared = rng.standard_normal((y.shape[0], dim))
        model, _ = fit_linear(x_shared[train_mask], y[train_mask])
        out[k] = r2_overall(y[test_mask], model.predict(x_shared[test_mask]))
    return out


def shuffle_targets(y, seed):
    """Permute the target rows against the entity IDs (alignment check)."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    return y[rng.permutation(y.shape[0])]


def popularity_features(urm_rows):
    """Interaction-count statistics of each entity's URM slice."""
    if sp.issparse(urm_rows):
        counts = np.asarray(urm_rows.sum(axis=1)).ravel()
        nnz = np.diff(urm_rows.tocsr().indptr).astype(np.float64)
    else:
        urm_rows = np.asarray(urm_rows, dtype=np.float64)
        counts = urm_rows.sum(axis=1)
        nnz = (urm_rows != 0).sum(axis=1).astype(np.float64)

    return np.column_stack([
        np.log1p(counts),
        np.log1p(nnz),
        counts,
        nnz,
    ])


def paired_effect_size(regressor_scores, baseline_scores, n_boot=10000, seed=0):
    """Effect size and CI for a paired regressor-vs-baseline comparison."""
    reg = np.asarray(regressor_scores, dtype=np.float64)
    base = np.asarray(baseline_scores, dtype=np.float64)
    ok = np.isfinite(reg) & np.isfinite(base)
    reg, base = reg[ok], base[ok]
    n = len(reg)
    if n == 0:
        return {"median_diff": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "cliffs_delta": float("nan"), "n_pairs": 0}

    diff = reg - base
    median_diff = float(np.median(diff))

    # Cliff's delta over the paired differences: the degree to which the
    # regressor exceeds the baseline, in [-1, 1].
    cliffs = float((np.sum(diff > 0) - np.sum(diff < 0)) / n)

    if n == 1 or np.allclose(diff, diff[0]):
        # A bootstrap over identical values is degenerate; the CI collapses to
        # the point estimate, which is the honest answer.
        return {"median_diff": median_diff, "ci_low": median_diff,
                "ci_high": median_diff, "cliffs_delta": cliffs, "n_pairs": n}

    rng = np.random.default_rng(seed)
    boot = np.median(rng.choice(diff, size=(n_boot, n), replace=True), axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"median_diff": median_diff, "ci_low": float(lo), "ci_high": float(hi),
            "cliffs_delta": cliffs, "n_pairs": n}


def paired_wilcoxon(regressor_scores, baseline_scores):
    """Paired Wilcoxon of the regressor vs a baseline over matched splits."""
    regressor_scores = np.asarray(regressor_scores, dtype=np.float64)
    baseline_scores = np.asarray(baseline_scores, dtype=np.float64)
    if np.allclose(regressor_scores, baseline_scores):
        return float("nan"), float("nan")
    stat, p = wilcoxon(regressor_scores, baseline_scores)
    return float(stat), float(p)
