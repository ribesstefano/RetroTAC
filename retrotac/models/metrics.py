"""
metrics.py
==========
Evaluation metrics shared by the nested-CV loop (`retrotac.models.training`).

Three groups, all operating on `(n_samples, n_targets)` arrays and averaging
per-target column, the same convention each model's own `.score()` already uses:

  * `regression_metrics`     -- R2/RMSE/MAE/... plus the rank correlations.
  * `classification_metrics` -- the regression problem read as a binary one by
    thresholding: labels come from `y_true >= threshold`, the label-based
    metrics (F1/MCC/...) from `y_pred >= threshold`, while the ranking metrics
    (ROC-AUC, PR-AUC) keep the *continuous* prediction as their score, since
    thresholding a score before feeding it to an AUC throws away exactly the
    information those metrics measure.
  * `hpo_objective`          -- the single scalar the Optuna inner loop
    MINIMISES: a convex blend of RMSE and a Spearman-derived rank penalty.

The target is assumed to live on [0, 1] (see `DEFAULT_CLF_THRESHOLD`).
"""
from typing import Dict, Iterator, Tuple

import numpy as np
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    explained_variance_score,
    f1_score,
    matthews_corrcoef,
    max_error,
    mean_absolute_error,
    median_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    root_mean_squared_error,
)

# Weight on RMSE in the composite HPO objective; (1 - alpha) goes to the rank
# penalty. 0.5 is nominal, not effective: on a [0, 1] target RMSE typically
# lands around 0.05-0.25 while the rank penalty spans the full [0, 1], so an
# alpha of 0.5 still lets ranking dominate. Tune it in models_config.yaml.
DEFAULT_OBJECTIVE_ALPHA = 0.5

# Cut applied to both the target and the predictions to derive binary labels.
DEFAULT_CLF_THRESHOLD = 0.7


def _as_2d(y: np.ndarray) -> np.ndarray:
    """Return `y` as a float 2-D array, promoting a flat vector to one column."""
    arr = np.asarray(y, dtype=float)
    return arr.reshape(-1, 1) if arr.ndim == 1 else arr


def _finite_columns(y_true: np.ndarray, y_pred: np.ndarray) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Yield each target column as a (true, pred) pair, dropping non-finite rows.

    A diverged model can emit NaN/inf predictions; those rows are dropped rather
    than poisoning every metric, so a partially-broken trial still gets scored on
    the part it did produce.

    Args:
        y_true: Ground-truth targets, shape (n_samples, n_targets) or (n_samples,).
        y_pred: Predictions, same shape as `y_true`.

    Yields:
        (col_true, col_pred) 1-D float arrays, restricted to finite pairs.

    Raises:
        ValueError: If the two arrays disagree on shape.
    """
    t2, p2 = _as_2d(y_true), _as_2d(y_pred)
    if t2.shape != p2.shape:
        raise ValueError(f"y_true {t2.shape} and y_pred {p2.shape} must have the same shape")
    for i in range(t2.shape[1]):
        t, p = t2[:, i], p2[:, i]
        mask = np.isfinite(t) & np.isfinite(p)
        yield t[mask], p[mask]


def _mean(values) -> float:
    """Mean across target columns, returning NaN for an empty list."""
    vals = list(values)
    return float(np.mean(vals)) if vals else float("nan")


def _safe_corr(fn, t: np.ndarray, p: np.ndarray) -> float:
    """Correlation coefficient from a scipy stat fn, NaN when it is undefined.

    Undefined here means fewer than two points or a constant input (scipy
    returns NaN and warns); callers decide how to treat that.
    """
    if t.size < 2 or np.ptp(t) == 0 or np.ptp(p) == 0:
        return float("nan")
    return float(fn(t, p)[0])


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Standard regression metrics, averaged over target columns.

    Args:
        y_true: Ground-truth targets, shape (n_samples, n_targets).
        y_pred: Predictions, same shape as `y_true`.

    Returns:
        Dict with r2, rmse, mae, medae, max_error, explained_variance, bias
        (mean signed error, pred - true) and the three rank/linear correlations
        pearson_r, spearman_rho, kendall_tau. Any metric that is undefined for
        the given data (e.g. a correlation on constant input) comes back as NaN.
    """
    cols = list(_finite_columns(y_true, y_pred))
    usable = [(t, p) for t, p in cols if t.size >= 2]
    return {
        "r2":                 _mean(r2_score(t, p) for t, p in usable),
        "rmse":               _mean(root_mean_squared_error(t, p) for t, p in usable),
        "mae":                _mean(mean_absolute_error(t, p) for t, p in usable),
        "medae":              _mean(median_absolute_error(t, p) for t, p in usable),
        "max_error":          _mean(max_error(t, p) for t, p in usable),
        "explained_variance": _mean(explained_variance_score(t, p) for t, p in usable),
        "bias":               _mean(np.mean(p - t) for t, p in usable),
        "pearson_r":          _mean(_safe_corr(pearsonr, t, p) for t, p in usable),
        "spearman_rho":       _mean(_safe_corr(spearmanr, t, p) for t, p in usable),
        "kendall_tau":        _mean(_safe_corr(kendalltau, t, p) for t, p in usable),
    }


def _binary_column_metrics(t: np.ndarray, p: np.ndarray, threshold: float) -> Dict[str, float]:
    """Binary metrics for one target column thresholded at `threshold`."""
    t_bin = (t >= threshold).astype(int)
    p_bin = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(t_bin, p_bin, labels=[0, 1]).ravel()
    both_classes = t_bin.min() != t_bin.max()      # AUCs need at least one of each
    return {
        "clf_roc_auc":           float(roc_auc_score(t_bin, p)) if both_classes else float("nan"),
        "clf_pr_auc":            float(average_precision_score(t_bin, p)) if both_classes else float("nan"),
        "clf_mcc":               float(matthews_corrcoef(t_bin, p_bin)),
        "clf_f1":                float(f1_score(t_bin, p_bin, zero_division=0)),
        "clf_precision":         float(precision_score(t_bin, p_bin, zero_division=0)),
        "clf_recall":            float(recall_score(t_bin, p_bin, zero_division=0)),
        "clf_specificity":       float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "clf_balanced_accuracy": float(balanced_accuracy_score(t_bin, p_bin)),
        "clf_accuracy":          float(accuracy_score(t_bin, p_bin)),
        "clf_cohen_kappa":       float(cohen_kappa_score(t_bin, p_bin)),
        "clf_pos_rate":          float(t_bin.mean()),        # PR-AUC's no-skill baseline
        "clf_pred_pos_rate":     float(p_bin.mean()),
        "clf_tp":                float(tp),
        "clf_fp":                float(fp),
        "clf_tn":                float(tn),
        "clf_fn":                float(fn),
    }


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = DEFAULT_CLF_THRESHOLD,
) -> Dict[str, float]:
    """Binary metrics obtained by thresholding the regression problem.

    Labels are `y_true >= threshold`; the label-based metrics use
    `y_pred >= threshold` while ROC-AUC/PR-AUC rank on the continuous `y_pred`.
    ROC-AUC and PR-AUC are NaN when a column's labels are single-class (a real
    possibility for a small scaffold fold at a strict threshold).

    Args:
        y_true: Ground-truth targets, shape (n_samples, n_targets).
        y_pred: Predictions, same shape as `y_true`.
        threshold: Cut on the target scale (default 0.7).

    Returns:
        Dict of `clf_*` metrics averaged over target columns, plus the
        `clf_threshold` used. Counts (tp/fp/tn/fn) are summed, not averaged.
    """
    cols = list(_finite_columns(y_true, y_pred))
    per_col = [_binary_column_metrics(t, p, threshold) for t, p in cols if t.size]
    if not per_col:
        return {"clf_threshold": float(threshold)}

    counts = {"clf_tp", "clf_fp", "clf_tn", "clf_fn"}
    out = {
        key: (float(sum(d[key] for d in per_col)) if key in counts
              else _mean(d[key] for d in per_col))
        for key in per_col[0]
    }
    out["clf_threshold"] = float(threshold)
    return out


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = DEFAULT_CLF_THRESHOLD,
) -> Dict[str, float]:
    """Regression + thresholded-classification metrics in one flat dict.

    Args:
        y_true: Ground-truth targets, shape (n_samples, n_targets).
        y_pred: Predictions, same shape as `y_true`.
        threshold: Binarisation cut forwarded to `classification_metrics`.

    Returns:
        The union of `regression_metrics` and `classification_metrics`; keys are
        disjoint (the binary ones are all `clf_`-prefixed).
    """
    return {
        **regression_metrics(y_true, y_pred),
        **classification_metrics(y_true, y_pred, threshold),
        "n_samples": float(_as_2d(y_true).shape[0]),
    }


def hpo_objective(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    alpha: float = DEFAULT_OBJECTIVE_ALPHA,
) -> Dict[str, float]:
    """Composite Optuna objective -- **lower is better**.

        objective = alpha * RMSE + (1 - alpha) * (1 - spearman_rho) / 2

    The rank penalty maps rho in [-1, 1] onto [0, 1], so a perfectly ordered
    prediction contributes 0 and a perfectly inverted one contributes 1. RMSE
    keeps the tuner honest about calibration, which a pure rank objective would
    happily throw away.

    A collapsed model (constant predictions, too few finite rows) yields an
    undefined rho; it is treated as rho = -1, the maximum penalty, rather than
    letting a NaN reach Optuna.

    Args:
        y_true: Ground-truth targets, shape (n_samples, n_targets).
        y_pred: Predictions, same shape as `y_true`.
        alpha: Weight on RMSE, in [0, 1]. `alpha=1` is pure RMSE, `alpha=0` is
            pure ranking. Must be a run-level constant, never an Optuna-suggested
            hyperparameter: trial values are only comparable under a fixed
            objective, and a tunable alpha just gets driven to whichever end
            flatters the trial at hand.

    Returns:
        Dict with `objective` plus its inputs `rmse`, `spearman_rho` and
        `rank_penalty`, so the trade-off can be inspected per trial.

    Raises:
        ValueError: If `alpha` is outside [0, 1].
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")

    usable = [(t, p) for t, p in _finite_columns(y_true, y_pred) if t.size >= 2]
    if not usable:
        # Nothing scoreable at all: worst plausible RMSE for a [0, 1] target
        # plus the maximum rank penalty.
        rmse, rho = 1.0, -1.0
    else:
        rmse = _mean(root_mean_squared_error(t, p) for t, p in usable)
        rhos = [_safe_corr(spearmanr, t, p) for t, p in usable]
        rho = _mean(-1.0 if np.isnan(r) else r for r in rhos)

    rank_penalty = (1.0 - rho) / 2.0
    return {
        "objective":    float(alpha * rmse + (1.0 - alpha) * rank_penalty),
        "rmse":         float(rmse),
        "spearman_rho": float(rho),
        "rank_penalty": float(rank_penalty),
    }
