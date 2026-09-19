"""compare_deeppsa_retrotac.py
=============================
Correlates DeepPSA's `es` (easy-to-synthesize probability) against RetroTAC's
real-valued ground truth and ensemble predictions, on the shared held-out
test set built by `prepare_test_set.py`.

DeepPSA's predictions CSV has three columns -- `es`, `hs`, `lable_pre` -- but
only `es` is used here; `hs`/`lable_pre` are ignored. `es` and `hs` are the
two softmax outputs of DeepPSA's classifier (`es + hs == 1`), so `es` alone
already carries everything `hs` would (`hs == 1 - es`) and `lable_pre` (0 =
easy, 1 = hard) is just DeepPSA's own argmax/threshold-at-0.5 call on that
same softmax, hence redundant too. `es` already uses RetroTAC's convention
(higher = easier to synthesize, same as `config/models_config.yaml`'s
`target: synthesizability`), so unlike the old categorical `lable_pre` it
needs no inversion before correlating against either side.

Pearson and Spearman are computed on `es`'s raw probability value. Kendall's
tau instead needs both sides binary, so it thresholds `es` at 0.5 (1 if
`es > 0.5` else 0) and separately thresholds the true `synthesizability`/
RetroTAC `prediction` side at `--threshold` (T, default 0.7, matching
`clf_threshold` in CLAUDE.md's metrics table) -- the two sides use different
cutoffs because they're different scales/conventions, not because of any
DeepPSA-specific reason.

Two correlations are always reported, both against `es`:
  1. true `synthesizability` (shared_test_set.csv) vs. DeepPSA `es`
  2. RetroTAC's `prediction` (retrotac_preds.csv) vs. DeepPSA `es`

Optionally, pass `--caruana-selection` pointing at `caruana_selection_smiles.csv`
(written by `scripts/models/evaluation.py`'s `run_ensemble_strategies`) to exclude,
before any correlation is computed, the rows of `--shared-test-set` that were used
to fit the Caruana ensemble weights -- those rows were "seen" during weight
fitting, so leaving them in would inflate the correlations reported for
RetroTAC's ensemble predictions.

Optionally, pass `--mol-scores` pointing at any external per-molecule scores
CSV (e.g. `data/deeppsa/routes_mol_heavy_scored.csv`, produced by
`retro_scores/synthesizability_scores.py`) to extend the analysis: every
numeric column in that CSV (or just `--mol-score-cols`, if given) is aligned
by SMILES against the true value, RetroTAC's prediction, and DeepPSA's `es`,
and the full pairwise correlation matrix (raw values throughout, no
thresholding) is written as both a CSV and a heatmap PNG. The external CSV
is typically a superset of/differently
ordered from the shared test set (e.g. it covers every scored route, not just
the DeepPSA-non-overlapping subset), so all four sources are inner-joined on
`smiles` -- never assumed row-aligned -- and any source rows that don't
survive the join are dropped and logged.

Whether or not `--mol-scores` is given, every log line (row-alignment stats,
correlation summaries, output paths) is also written to a `.log` file in
`--output-dir`, not just the console.

The three required CSVs share the same molecules but *not* the same row
order (DeepPSA's own pipeline reorders its output on load), so rows are
joined on `smiles` string equality rather than position -- verified exact
1:1 match between their SMILES sets, so no rdkit standardization is needed
here (contrast `prepare_test_set.py`, which does need it).

Needs pandas/scipy/numpy/matplotlib/seaborn -- apptainer/training.sif has all
of these (see CLAUDE.md "Containers"):
    apptainer exec $(bash apptainer/bind_live_repo.sh) apptainer/training.sif \\
        python scripts/deeppsa/compare_deeppsa_retrotac.py

Usage
-----
    python scripts/deeppsa/compare_deeppsa_retrotac.py \\
        --deeppsa-preds data/deeppsa/deeppsa_preds.csv \\
        --shared-test-set data/deeppsa/shared_test_set/shared_test_set.csv \\
        --retrotac-preds data/deeppsa/retrotac_preds.csv \\
        --caruana-selection outputs/results/results_20260828_182305/caruana_selection_smiles.csv \\
        --mol-scores data/deeppsa/routes_mol_heavy_scored.csv \\
        --output-dir outputs/deeppsa
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import kendalltau, pearsonr, spearmanr

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# DeepPSA's `es` column, copied under this name. Not underscore-prefixed like
# an internal-only column: it ends up as an axis label on the
# correlation-matrix heatmap when --mol-scores is given.
_DEEPPSA_PRED_COL = "deeppsa_es"


def _add_file_handler(output_dir: Path) -> Path:
    """(Re-)attach a file handler so this run's log output is persisted to disk.

    Drops any file handler left by an earlier call in the same process (e.g.
    repeated calls from a notebook) first, so log files never accumulate
    duplicate lines across runs.
    """
    for h in list(logger.handlers):
        if isinstance(h, logging.FileHandler):
            logger.removeHandler(h)
            h.close()
    log_path = output_dir / "compare_deeppsa_retrotac.log"
    handler = logging.FileHandler(log_path, mode="w")
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    return log_path


def _correlations(x_raw: np.ndarray, y_raw: np.ndarray, x_bin: np.ndarray, y_bin: np.ndarray) -> Dict[str, float]:
    """Pearson/Spearman on the raw pair, Kendall's tau on a separately-thresholded binary pair.

    Pearson/Spearman use `x_raw`/`y_raw` directly. Kendall's tau uses
    `x_bin`/`y_bin` instead: DeepPSA's `es` is thresholded at 0.5 and the
    true/RetroTAC side at `--threshold` (see module docstring), each on a
    different scale, so the raw and binary pairs are computed and
    variance-checked independently -- thresholding can zero out variance the
    raw values had, and vice versa.

    Mirrors `retrotac.models.metrics._safe_corr`'s guard: NaN (rather than a
    scipy warning) for fewer than two points or a constant input.
    """
    result = {"pearson_r": float("nan"), "spearman_rho": float("nan"), "kendall_tau": float("nan")}
    if x_raw.size >= 2 and np.ptp(x_raw) != 0 and np.ptp(y_raw) != 0:
        result["pearson_r"] = float(pearsonr(x_raw, y_raw)[0])
        result["spearman_rho"] = float(spearmanr(x_raw, y_raw)[0])
    if x_bin.size >= 2 and np.ptp(x_bin) != 0 and np.ptp(y_bin) != 0:
        result["kendall_tau"] = float(kendalltau(x_bin, y_bin)[0])
    return result


def _exclude_caruana_selection(
    shared_test_set: pd.DataFrame, caruana_selection_path: Path, smiles_col: str,
) -> pd.DataFrame:
    """Drop rows of `shared_test_set` whose SMILES appear in the Caruana selection CSV.

    Args:
        shared_test_set: Ground-truth frame to filter; must have `smiles_col`.
        caruana_selection_path: `caruana_selection_smiles.csv` written by
            `scripts/models/evaluation.py`'s `run_ensemble_strategies` -- always
            has a `smiles` column (that CSV's schema is fixed by its writer,
            independent of `smiles_col`).
        smiles_col: SMILES column name in `shared_test_set`.

    Returns:
        `shared_test_set` with selection-set rows removed.
    """
    selection_smiles = set(pd.read_csv(caruana_selection_path)["smiles"])
    n_before = len(shared_test_set)
    filtered = shared_test_set[~shared_test_set[smiles_col].isin(selection_smiles)]
    logger.info(
        "Caruana selection filter: excluded %d/%d shared-test-set rows present in %s; %d rows remain",
        n_before - len(filtered), n_before, caruana_selection_path, len(filtered),
    )
    return filtered


def _merge_on_smiles(left: pd.DataFrame, right: pd.DataFrame, right_name: str, smiles_col: str) -> pd.DataFrame:
    """Inner-join `left` and `right` on `smiles_col`, 1:1, logging any drop."""
    merged = left.merge(right, on=smiles_col, how="inner", validate="one_to_one")
    n_dropped = len(left) - len(merged)
    if n_dropped:
        logger.warning("%d/%d rows had no %s match on %r and were dropped",
                       n_dropped, len(left), right_name, smiles_col)
    if merged.empty:
        raise ValueError(f"no overlapping {smiles_col!r} values against {right_name}")
    return merged


def _score_columns(mol_scores: pd.DataFrame, smiles_col: str, requested: Optional[List[str]]) -> List[str]:
    """Molecular-score columns to correlate: `requested` if given, else every
    numeric column in `mol_scores` other than `smiles_col`.
    """
    if requested:
        missing = [c for c in requested if c not in mol_scores.columns]
        if missing:
            raise ValueError(f"--mol-score-cols not found in the molecular-scores CSV: {missing}")
        return requested
    numeric = mol_scores.drop(columns=[smiles_col]).select_dtypes(include="number").columns.tolist()
    if not numeric:
        raise ValueError(f"no numeric columns found in the molecular-scores CSV besides {smiles_col!r}")
    return numeric


def _drop_degenerate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns that are all-NaN or constant -- undefined/uninformative in a correlation matrix."""
    keep = [c for c in df.columns if df[c].notna().sum() >= 2 and np.ptp(df[c].dropna()) > 0]
    dropped = [c for c in df.columns if c not in keep]
    if dropped:
        logger.warning("dropping degenerate columns from the correlation matrix (all-NaN or constant): %s", dropped)
    return df[keep]


def _plot_correlation_matrix(matrix: pd.DataFrame, output_path: Path, method: str) -> None:
    """Render `matrix` as an annotated heatmap and save it to `output_path`."""
    n = len(matrix)
    fig, ax = plt.subplots(figsize=(max(8.0, 0.6 * n), max(6.5, 0.6 * n)))
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1,
               square=True, linewidths=0.5, cbar_kws={"label": f"{method} correlation"}, ax=ax)
    ax.set_title(f"{method.capitalize()} correlation matrix")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _molecular_scores_analysis(
    deeppsa: pd.DataFrame,
    shared_test_set: pd.DataFrame,
    retrotac: pd.DataFrame,
    mol_scores_path: Path,
    output_dir: Path,
    smiles_col: str,
    target_col: str,
    prediction_col: str,
    mol_score_cols: Optional[List[str]],
    corr_method: str,
) -> Dict[str, Any]:
    """Align DeepPSA/RetroTAC/ground-truth with an external molecular-scores CSV
    on `smiles_col`, then compute and plot a full pairwise correlation matrix.

    Args:
        deeppsa: DeepPSA predictions with `_DEEPPSA_PRED_COL` already attached
            (see `compare_deeppsa_retrotac`).
        shared_test_set: Ground-truth frame (must have `smiles_col`, `target_col`).
        retrotac: RetroTAC predictions frame (must have `smiles_col`, `prediction_col`).
        mol_scores_path: External per-molecule scores CSV (must have `smiles_col`);
            typically a superset of the shared test set, in unrelated row order.
        output_dir: Directory to write `correlation_matrix.csv`/`.png` into.
        smiles_col: SMILES column name, shared by every input.
        target_col: Ground-truth column name.
        prediction_col: RetroTAC prediction column name.
        mol_score_cols: Explicit molecular-score columns to use, or None to use
            every numeric column in `mol_scores_path` besides `smiles_col`.
        corr_method: One of "pearson"/"spearman"/"kendall".

    Returns:
        Summary dict: `method`, `n_samples`, `columns`, `vs_true`,
        `vs_prediction`, `vs_deeppsa` (each of the last three a
        `{column: correlation}` dict, sliced straight from the saved matrix),
        plus the `matrix_csv`/`matrix_png` output paths.
    """
    mol_scores = pd.read_csv(mol_scores_path)
    score_cols = _score_columns(mol_scores, smiles_col, mol_score_cols)
    logger.info("molecular scores (%s): %d rows, %d score columns: %s",
               mol_scores_path, len(mol_scores), len(score_cols), score_cols)

    merged = deeppsa[[smiles_col, _DEEPPSA_PRED_COL]]
    merged = _merge_on_smiles(merged, shared_test_set[[smiles_col, target_col]], "shared_test_set", smiles_col)
    merged = _merge_on_smiles(merged, retrotac[[smiles_col, prediction_col]], "retrotac_preds", smiles_col)
    merged = _merge_on_smiles(merged, mol_scores[[smiles_col] + score_cols], "mol_scores", smiles_col)
    logger.info("aligned on %r across true/RetroTAC/DeepPSA/mol-scores: %d rows remain (of %d mol-scores rows)",
               smiles_col, len(merged), len(mol_scores))

    numeric = _drop_degenerate_columns(merged.drop(columns=[smiles_col]))
    matrix = numeric.corr(method=corr_method)

    matrix_csv_path = output_dir / "correlation_matrix.csv"
    matrix_png_path = output_dir / "correlation_matrix.png"
    matrix.to_csv(matrix_csv_path)
    _plot_correlation_matrix(matrix, matrix_png_path, corr_method)
    logger.info("correlation matrix (%s, %d variables) written to %s and %s",
               corr_method, len(matrix), matrix_csv_path, matrix_png_path)

    kept_score_cols = [c for c in score_cols if c in matrix.columns]
    vs_true = matrix.loc[kept_score_cols, target_col].to_dict() if target_col in matrix.columns else {}
    vs_prediction = matrix.loc[kept_score_cols, prediction_col].to_dict() if prediction_col in matrix.columns else {}
    vs_deeppsa = matrix.loc[kept_score_cols, _DEEPPSA_PRED_COL].to_dict() if _DEEPPSA_PRED_COL in matrix.columns else {}

    logger.info("%s correlation of each molecular score vs. true/RetroTAC/DeepPSA (n=%d):", corr_method, len(merged))
    for col in kept_score_cols:
        logger.info("  %-20s vs_true=%7.4f vs_prediction=%7.4f vs_deeppsa=%7.4f",
                   col, vs_true.get(col, float("nan")), vs_prediction.get(col, float("nan")), vs_deeppsa.get(col, float("nan")))

    return {
        "method": corr_method,
        "n_samples": len(merged),
        "columns": kept_score_cols,
        "vs_true": vs_true,
        "vs_prediction": vs_prediction,
        "vs_deeppsa": vs_deeppsa,
        "matrix_csv": str(matrix_csv_path),
        "matrix_png": str(matrix_png_path),
    }


def compare_deeppsa_retrotac(
    deeppsa_preds_path: Path,
    shared_test_set_path: Path,
    retrotac_preds_path: Path,
    output_dir: Path,
    smiles_col: str = "smiles",
    deeppsa_score_col: str = "es",
    target_col: str = "synthesizability",
    prediction_col: str = "prediction",
    threshold: float = 0.7,
    caruana_selection_path: Optional[Path] = None,
    mol_scores_path: Optional[Path] = None,
    mol_score_cols: Optional[List[str]] = None,
    corr_method: str = "spearman",
) -> Dict[str, Any]:
    """Correlate DeepPSA's `es` probability against RetroTAC, and optionally
    against an external molecular-scores CSV.

    Args:
        deeppsa_preds_path: DeepPSA's per-molecule predictions CSV; must have
            `smiles_col` and `deeppsa_score_col` (the easy-to-synthesize
            softmax probability, in [0, 1]; `hs`/`lable_pre` are ignored,
            see module docstring).
        shared_test_set_path: The shared held-out set built by
            `prepare_test_set.py`; must have `smiles_col` and `target_col`
            (the real-valued ground truth RetroTAC is trained to predict).
        retrotac_preds_path: RetroTAC ensemble predictions on the same set;
            must have `smiles_col` and `prediction_col`.
        output_dir: Directory for `correlation_results.json`, the `.log` file,
            and (only if `mol_scores_path` is given) `correlation_matrix.csv`/
            `.png`. Created if missing.
        smiles_col: SMILES column name, shared by all inputs.
        deeppsa_score_col: DeepPSA's easy-to-synthesize probability column.
        target_col: Real-valued ground-truth column in `shared_test_set_path`.
        prediction_col: RetroTAC's real-valued prediction column in
            `retrotac_preds_path`.
        threshold: Cutoff T used only for Kendall's tau's binary pair: true
            `synthesizability`/RetroTAC `prediction` are thresholded at T
            (1 if > T else 0), while DeepPSA `es` is always thresholded at a
            fixed 0.5 regardless of T (see module docstring). Pearson/Spearman
            are unaffected and always use raw values.
        caruana_selection_path: Optional `caruana_selection_smiles.csv` (see
            module docstring). When given, rows of `shared_test_set_path`
            whose SMILES appear there are dropped before any merge/
            correlation, so they affect every reported correlation (the two
            base ones and, if `mol_scores_path` is also given, the extended
            matrix).
        mol_scores_path: Optional external per-molecule scores CSV. When
            given, triggers the extended SMILES-aligned correlation-matrix
            analysis (see `_molecular_scores_analysis`).
        mol_score_cols: Explicit columns to use from `mol_scores_path`, or
            None to auto-detect every numeric column besides `smiles_col`.
            Ignored when `mol_scores_path` is None.
        corr_method: Correlation method for the `--mol-scores` matrix -- one
            of "pearson"/"spearman"/"kendall", computed on raw values
            throughout (the Kendall-only thresholding above applies solely to
            the two base comparisons, not this matrix). Ignored when
            `mol_scores_path` is None.

    Returns:
        `{"true_vs_deeppsa": {...}, "retrotac_vs_deeppsa": {...}}`, each an
        inner dict of `pearson_r`/`spearman_rho`/`kendall_tau`/`n_samples`,
        plus `"n_excluded_caruana_selection"` (only when `caruana_selection_path`
        is given) and `"molecular_scores": {...}` (see
        `_molecular_scores_analysis`) when `mol_scores_path` is given.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = _add_file_handler(output_dir)

    deeppsa = pd.read_csv(deeppsa_preds_path)
    shared_test_set = pd.read_csv(shared_test_set_path)
    retrotac = pd.read_csv(retrotac_preds_path)

    n_excluded_caruana_selection = None
    if caruana_selection_path is not None:
        n_before = len(shared_test_set)
        shared_test_set = _exclude_caruana_selection(shared_test_set, caruana_selection_path, smiles_col)
        n_excluded_caruana_selection = n_before - len(shared_test_set)

    # `es` already uses RetroTAC's higher-means-easier convention (see module
    # docstring), so it's copied as-is -- no inversion needed.
    deeppsa[_DEEPPSA_PRED_COL] = deeppsa[deeppsa_score_col].astype(float)

    merged = deeppsa[[smiles_col, _DEEPPSA_PRED_COL]]
    merged = _merge_on_smiles(merged, shared_test_set[[smiles_col, target_col]], "shared_test_set", smiles_col)
    merged = _merge_on_smiles(merged, retrotac[[smiles_col, prediction_col]], "retrotac_preds", smiles_col)

    es = merged[_DEEPPSA_PRED_COL].to_numpy(dtype=float)
    es_bin = (es > 0.5).astype(float)
    true_raw = merged[target_col].to_numpy(dtype=float)
    true_bin = (true_raw > threshold).astype(float)
    pred_raw = merged[prediction_col].to_numpy(dtype=float)
    pred_bin = (pred_raw > threshold).astype(float)

    corr_true = _correlations(true_raw, es, true_bin, es_bin)
    corr_true["n_samples"] = len(merged)
    corr_pred = _correlations(pred_raw, es, pred_bin, es_bin)
    corr_pred["n_samples"] = len(merged)

    results: Dict[str, Any] = {"true_vs_deeppsa": corr_true, "retrotac_vs_deeppsa": corr_pred}
    if n_excluded_caruana_selection is not None:
        results["n_excluded_caruana_selection"] = n_excluded_caruana_selection

    logger.info("true %s vs DeepPSA %s (n=%d): pearson_r=%.4f spearman_rho=%.4f kendall_tau=%.4f (es>0.5 vs %s>%.2f)",
               target_col, deeppsa_score_col, corr_true["n_samples"], corr_true["pearson_r"], corr_true["spearman_rho"],
               corr_true["kendall_tau"], target_col, threshold)
    logger.info("RetroTAC %s vs DeepPSA %s (n=%d): pearson_r=%.4f spearman_rho=%.4f kendall_tau=%.4f (es>0.5 vs %s>%.2f)",
               prediction_col, deeppsa_score_col, corr_pred["n_samples"], corr_pred["pearson_r"], corr_pred["spearman_rho"],
               corr_pred["kendall_tau"], prediction_col, threshold)

    if mol_scores_path is not None:
        results["molecular_scores"] = _molecular_scores_analysis(
            deeppsa, shared_test_set, retrotac, mol_scores_path, output_dir,
            smiles_col, target_col, prediction_col, mol_score_cols, corr_method,
        )

    results_path = output_dir / "correlation_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("results written to %s", results_path)
    logger.info("log written to %s", log_path)

    return results


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the correlation CLI.

    Returns:
        Parsed arguments namespace.
    """
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--deeppsa-preds", type=Path,
                    default=_PROJECT_ROOT / "data" / "deeppsa" / "deeppsa_preds.csv",
                    help="DeepPSA's per-molecule predictions CSV (default: %(default)s)")
    ap.add_argument("--shared-test-set", type=Path,
                    default=_PROJECT_ROOT / "data" / "deeppsa" / "shared_test_set" / "shared_test_set.csv",
                    help="shared held-out test set with ground truth (default: %(default)s)")
    ap.add_argument("--retrotac-preds", type=Path,
                    default=_PROJECT_ROOT / "data" / "deeppsa" / "retrotac_preds.csv",
                    help="RetroTAC ensemble predictions on the shared test set (default: %(default)s)")
    ap.add_argument("--caruana-selection", type=Path, default=None,
                    help="optional caruana_selection_smiles.csv (written by scripts/models/"
                         "evaluation.py's run_ensemble_strategies); when given, --shared-test-set "
                         "rows whose SMILES appear there are excluded before any correlation is "
                         "computed (default: none)")
    ap.add_argument("--mol-scores", type=Path, default=None,
                    help="optional external per-molecule scores CSV (e.g. "
                         "data/deeppsa/routes_mol_heavy_scored.csv); when given, also aligns it by "
                         "SMILES and writes a full correlation matrix (CSV + heatmap PNG) (default: none)")
    ap.add_argument("--mol-score-cols", default=None,
                    help="comma-separated columns to use from --mol-scores "
                         "(default: every numeric column except --smiles-col)")
    ap.add_argument("--corr-method", choices=["pearson", "spearman", "kendall"], default="spearman",
                    help="correlation method for the --mol-scores matrix (default: %(default)s)")
    ap.add_argument("--output-dir", type=Path,
                    default=_PROJECT_ROOT / "outputs" / "deeppsa",
                    help="output directory for the results JSON, log file, and (if --mol-scores is "
                         "given) the correlation matrix (default: %(default)s)")
    ap.add_argument("--smiles-col", default="smiles",
                    help="SMILES column name shared by all inputs (default: %(default)s)")
    ap.add_argument("--deeppsa-score-col", default="es",
                    help="DeepPSA's easy-to-synthesize probability column (`hs`/`lable_pre` are "
                         "ignored, see module docstring) (default: %(default)s)")
    ap.add_argument("--target-col", default="synthesizability",
                    help="real-valued ground-truth column in --shared-test-set (default: %(default)s)")
    ap.add_argument("--prediction-col", default="prediction",
                    help="RetroTAC's real-valued prediction column in --retrotac-preds (default: %(default)s)")
    ap.add_argument("--threshold", type=float, default=0.7,
                    help="cutoff T used only for Kendall's tau: true synthesizability/RetroTAC "
                         "prediction are thresholded at T, DeepPSA --deeppsa-score-col always at a "
                         "fixed 0.5 (default: %(default)s)")
    return ap.parse_args()


def main() -> None:
    """CLI entry point: parse arguments and run the correlation comparison."""
    args = parse_args()
    mol_score_cols = [c.strip() for c in args.mol_score_cols.split(",")] if args.mol_score_cols else None
    compare_deeppsa_retrotac(
        args.deeppsa_preds, args.shared_test_set, args.retrotac_preds, args.output_dir,
        smiles_col=args.smiles_col, deeppsa_score_col=args.deeppsa_score_col,
        target_col=args.target_col, prediction_col=args.prediction_col, threshold=args.threshold,
        caruana_selection_path=args.caruana_selection,
        mol_scores_path=args.mol_scores, mol_score_cols=mol_score_cols, corr_method=args.corr_method,
    )


if __name__ == "__main__":
    main()
