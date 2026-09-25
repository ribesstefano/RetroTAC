"""
scripts/plotting_appendix/hpo_details.py
========================================
What the inner Optuna loop searched, what it selected, and how far it got,
for the appendix's hyperparameter-optimisation section.

Three artifacts are read, all written by `train.py` per outer fold under
`outputs/cv/<run>/`:

  * `trials_seed*_fold*.csv` -- every trial's parameters, the composite inner
    objective (minimised) and its `rmse` / `spearman_rho` user attributes.
    Only COMPLETE trials enter a study's best-so-far trace, since Optuna's
    own `best_trial` ignores pruned ones.
  * `best_params_seed*_fold*.json` -- the configuration each of the 25 outer
    folds refit on, plus the inner objective it achieved.
  * `outputs/models/<run>/<run>_hparams.yaml` -- the single configuration the
    final model (refit on the whole development set) uses.

The selection table reports the spread of each hyperparameter over the 25
folds rather than a single number: an interval that spans most of its search
range says the objective is flat in that direction, which is a different
statement from "this is the value to use".

Usage
-----
    python scripts/plotting_appendix/hpo_details.py \\
        --runs xgb_20260828_182305 mlp_20260828_182305 gnn_20260828_182305

Outputs (see --out-dir / --fig-dir):
    outputs/appendix/hpo_selected_params.csv
    outputs/appendix/hpo_study_summary.csv
    figures/appendix/hpo_search.{pdf,svg,png}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

import pubstyle as ps

ps.apply_style()

#: Backend -> hue, matching scripts/models/plotting_evaluation.py.
COLOR_MAP = {
    "XGB": ps.PALETTE["blue"],
    "MLP": ps.PALETTE["purple"],
    "GNN": ps.PALETTE["dark_orange"],
}

#: Keys `best_params_*.json` carries that record the objective rather than a
#: searched hyperparameter.
OBJECTIVE_KEYS = ("best_objective_inner", "objective_alpha", "inner_rmse",
                  "inner_spearman_rho", "inner_rank_penalty")

#: Hyperparameters searched on a log scale, reported as geometric rather than
#: arithmetic summaries so the interval is symmetric in the searched space.
LOG_SCALE = ("learning_rate", "max_lr", "weight_decay")

#: Hyperparameters drawn from a fixed set of levels, summarised by how often
#: each level was selected rather than by a median.
CATEGORICAL = ("hidden_dim", "ffn_hidden_dim", "n_layers", "ffn_n_layers",
               "max_depth", "min_child_weight", "batch_norm")


def load_study(cv_dir: Path) -> pd.DataFrame:
    """Stack every outer fold's Optuna trial table for one run.

    Args:
        cv_dir: `outputs/cv/<run>`.

    Returns:
        Long frame with a `seed` and `fold` column added per source file.
    """
    frames = []
    for path in sorted(cv_dir.glob("trials_seed*_fold*.csv")):
        stem = path.stem.replace("trials_seed", "")
        seed, fold = stem.split("_fold")
        frames.append(pd.read_csv(path).assign(seed=int(seed), fold=int(fold)))
    if not frames:
        raise SystemExit(f"no trial tables under {cv_dir}")
    return pd.concat(frames, ignore_index=True)


def load_best_params(cv_dir: Path) -> pd.DataFrame:
    """Stack the per-fold selected configurations for one run.

    Args:
        cv_dir: `outputs/cv/<run>`.

    Returns:
        One row per (seed, fold).
    """
    rows = []
    for path in sorted(cv_dir.glob("best_params_seed*_fold*.json")):
        stem = path.stem.replace("best_params_seed", "")
        seed, fold = stem.split("_fold")
        rows.append({"seed": int(seed), "fold": int(fold),
                     **json.loads(path.read_text())})
    if not rows:
        raise SystemExit(f"no best-params files under {cv_dir}")
    return pd.DataFrame(rows)


def summarise_selection(best: pd.DataFrame, model: str) -> pd.DataFrame:
    """Spread of each selected hyperparameter across the 25 outer folds.

    Args:
        best: Output of `load_best_params`.
        model: Display name, carried into the output.

    Returns:
        One row per hyperparameter, with a median and an interquartile range
        for numeric parameters and a modal level plus its frequency for
        categorical ones.
    """
    rows = []
    for column in best.columns:
        if column in ("seed", "fold") or column in OBJECTIVE_KEYS:
            continue
        values = best[column].dropna()
        if values.empty:
            continue
        row = {"model": model, "param": column, "n_folds": int(len(values))}
        if column in CATEGORICAL or values.dtype == object or values.dtype == bool:
            counts = values.value_counts()
            row["summary"] = f"{counts.index[0]} ({counts.iloc[0]}/{len(values)})"
            row["levels"] = "; ".join(f"{k}: {v}" for k, v in counts.items())
            row["median"] = np.nan
        else:
            numeric = values.astype(float)
            q1, q2, q3 = np.percentile(numeric, [25, 50, 75])
            row["median"] = float(q2)
            row["q1"], row["q3"] = float(q1), float(q3)
            if column in LOG_SCALE:
                row["summary"] = f"{q2:.2e} [{q1:.2e}, {q3:.2e}]"
            elif float(numeric.max()) - float(numeric.min()) < 1 and numeric.max() <= 1:
                row["summary"] = f"{q2:.3f} [{q1:.3f}, {q3:.3f}]"
            else:
                row["summary"] = f"{q2:.4g} [{q1:.4g}, {q3:.4g}]"
        rows.append(row)
    return pd.DataFrame(rows)


def summarise_study(trials: pd.DataFrame, best: pd.DataFrame, model: str) -> Dict:
    """Budget, pruning rate and achieved inner objective for one run.

    Args:
        trials: Output of `load_study`.
        best: Output of `load_best_params`.
        model: Display name.

    Returns:
        One summary record.
    """
    per_study = trials.groupby(["seed", "fold"]).size()
    complete = trials["state"].eq("COMPLETE")
    return {
        "model": model,
        "n_studies": int(per_study.size),
        "trials_per_study": int(per_study.max()),
        "trials_total": int(len(trials)),
        "pruned_pct": float(100.0 * (~complete).mean()),
        "best_objective_mean": float(best["best_objective_inner"].mean()),
        "best_objective_std": float(best["best_objective_inner"].std(ddof=1)),
        "inner_rmse_mean": float(best["inner_rmse"].mean()),
        "inner_spearman_mean": float(best["inner_spearman_rho"].mean()),
    }


def best_so_far(trials: pd.DataFrame) -> pd.DataFrame:
    """Running minimum of the inner objective, per study, by trial number.

    Pruned trials never become a study's incumbent, so they are excluded from
    the running minimum while still consuming a trial number -- which is what
    makes the MLP's curve flatten later than its completed-trial count alone
    would suggest. The incumbent is carried forward across those gaps, and
    back-filled over any leading pruned trials, so each study's trace is
    monotone from trial 1 and the median across studies is monotone too.

    Args:
        trials: Output of `load_study`.

    Returns:
        Frame indexed by trial number with one column per (seed, fold) study.
    """
    columns = {}
    for (seed, fold), group in trials.groupby(["seed", "fold"]):
        group = group.sort_values("number")
        values = group["value"].where(group["state"].eq("COMPLETE"))
        columns[f"{seed}_{fold}"] = values.cummin().ffill().bfill().to_numpy()
    width = max(len(v) for v in columns.values())
    padded = {k: np.concatenate([v, np.full(width - len(v), np.nan)])
              for k, v in columns.items()}
    return pd.DataFrame(padded, index=np.arange(1, width + 1))


def plot_search(traces: Dict[str, pd.DataFrame], trials: Dict[str, pd.DataFrame],
                out_stem: Path) -> List[Path]:
    """Convergence of the inner objective and the trade-off it optimises.

    Args:
        traces: Model name -> output of `best_so_far`.
        trials: Model name -> output of `load_study`.
        out_stem: Output path without extension.

    Returns:
        Paths written.
    """
    width, _ = ps.set_size()
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(width, width * 0.40),
                                     layout="constrained")

    for model, trace in traces.items():
        median = trace.median(axis=1, skipna=True)
        lo = trace.quantile(0.25, axis=1)
        hi = trace.quantile(0.75, axis=1)
        ax_a.plot(trace.index, median, color=COLOR_MAP[model], label=model)
        ax_a.fill_between(trace.index, lo, hi, color=COLOR_MAP[model],
                          alpha=0.20, linewidth=0)
    ax_a.set_xlabel("Optuna trial index")
    ax_a.set_ylabel("Best inner objective so far")
    ax_a.legend(loc="lower right")
    ax_a.tick_params(axis="both", which="major", labelsize=ps.ANNOT_FONTSIZE)
    ax_a.grid(False)

    # The composite objective trades calibration against ordering; plotting
    # the two attributes it is built from shows what the search actually
    # explored, and that the backends occupy different regions of it.
    for model, frame in trials.items():
        done = frame[frame["state"].eq("COMPLETE")]
        ax_b.scatter(done["user_attrs_spearman_rho"], done["user_attrs_rmse"],
                     s=5, alpha=0.45, linewidths=0, color=COLOR_MAP[model],
                     label=model, rasterized=True)
    ax_b.set_xlabel("Inner-validation Spearman $\\rho$")
    ax_b.set_ylabel("Inner-validation RMSE")
    ax_b.legend(loc="lower left", markerscale=2.0)
    ax_b.tick_params(axis="both", which="major", labelsize=ps.ANNOT_FONTSIZE)
    ax_b.grid(False)

    for ax, tag in ((ax_a, "(a)"), (ax_b, "(b)")):
        ax.text(-0.17, 1.07, tag, transform=ax.transAxes, fontweight="bold",
                fontsize=ps.LABEL_FONTSIZE, va="top", ha="left", clip_on=False)
    return ps.save_figure(fig, out_stem)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed namespace.
    """
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cv-root", type=Path, default=Path("outputs/cv"),
                   help="Directory holding <run>/ with the per-fold Optuna artifacts.")
    p.add_argument("--models-root", type=Path, default=Path("outputs/models"),
                   help="Directory holding <run>/<run>_hparams.yaml.")
    p.add_argument("--runs", nargs="+", default=[
        "xgb_20260828_182305", "mlp_20260828_182305", "gnn_20260828_182305"],
        help="Run identifiers, in display order.")
    p.add_argument("--out-dir", type=Path, default=Path("outputs/appendix"),
                   help="Directory for the tables.")
    p.add_argument("--fig-dir", type=Path, default=Path("figures/appendix"),
                   help="Directory for the figure.")
    return p.parse_args()


def main() -> None:
    """Write the HPO tables and the search figure."""
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    traces, all_trials, selections, summaries, finals = {}, {}, [], [], {}
    for run in args.runs:
        model = run.split("_")[0].upper()
        cv_dir = args.cv_root / run
        trials = load_study(cv_dir)
        best = load_best_params(cv_dir)
        all_trials[model] = trials
        traces[model] = best_so_far(trials)
        selections.append(summarise_selection(best, model))
        summaries.append(summarise_study(trials, best, model))
        hparams_path = args.models_root / run / f"{run}_hparams.yaml"
        if hparams_path.exists():
            finals[model] = yaml.safe_load(hparams_path.read_text())["hyperparameters"]

    summary = pd.DataFrame(summaries)
    summary.to_csv(args.out_dir / "hpo_study_summary.csv", index=False)
    print("=== per-backend search budget and outcome ===")
    print(summary.round(4).to_string(index=False))

    selection = pd.concat(selections, ignore_index=True)
    # Attach the value the final full-development-set model uses, beside the
    # spread of what the 25 folds picked.
    selection["final_model_value"] = [
        finals.get(row.model, {}).get(row.param, "") for row in selection.itertuples()]
    selection.to_csv(args.out_dir / "hpo_selected_params.csv", index=False)
    print("\n=== selected hyperparameters: median [IQR] over 25 outer folds ===")
    print(selection[["model", "param", "summary", "final_model_value"]]
          .to_string(index=False))

    written = plot_search(traces, all_trials, args.fig_dir / "hpo_search")
    print("\nwrote:", *[str(p) for p in written], sep="\n  ")


if __name__ == "__main__":
    main()
