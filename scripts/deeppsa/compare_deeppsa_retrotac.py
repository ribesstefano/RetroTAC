"""compare_deeppsa_retrotac.py
=============================
Correlates DeepPSA's categorical synthesizability call against RetroTAC's
real-valued ground truth and ensemble predictions, on the shared held-out
test set built by `prepare_test_set.py`.

DeepPSA and RetroTAC use opposite label conventions: DeepPSA's `lable_pre`
column is 0 = easy, 1 = hard, while RetroTAC's target/predictions live on
[0, 1] with 0 = hard, 1 = easy (`config/models_config.yaml`'s
`target: synthesizability`). `lable_pre` is inverted (`1 - lable_pre`) before
any correlation is computed, so both sides agree that higher means easier.
Two correlations are always reported, both against this inverted DeepPSA call:
  1. true `synthesizability` (shared_test_set.csv) vs. inverted DeepPSA label
  2. RetroTAC's `prediction` (retrotac_preds.csv) vs. inverted DeepPSA label

Optionally, pass `--mol-scores` pointing at any external per-molecule scores
CSV (e.g. `data/deeppsa/routes_mol_heavy_scored.csv`, produced by
`retro_scores/synthesizability_scores.py`) to extend the analysis: every
numeric column in that CSV (or just `--mol-score-cols`, if given) is aligned
by SMILES against the true value, RetroTAC's prediction, and the inverted
DeepPSA label, and the full pairwise correlation matrix is written as both a
CSV and a heatmap PNG. The external CSV is typically a superset of/differently
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

# DeepPSA's `lable_pre` inverted to RetroTAC's convention (0=hard, 1=easy).
# Not underscore-prefixed like an internal-only column: it ends up as an axis
# label on the correlation-matrix heatmap when --mol-scores is given.
_DEEPPSA_PRED_COL = "deeppsa_pred_easy"


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


def _correlations(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """Pearson/Spearman/Kendall correlation between two 1-D arrays.

    Mirrors `retrotac.models.metrics._safe_corr`'s guard: NaN (rather than a
    scipy warning) for fewer than two points or a constant input.
    """
    if x.size < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return {"pearson_r": float("nan"), "spearman_rho": float("nan"), "kendall_tau": float("nan")}
    return {
        "pearson_r": float(pearsonr(x, y)[0]),
        "spearman_rho": float(spearmanr(x, y)[0]),
        "kendall_tau": float(kendalltau(x, y)[0]),
    }


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
    deeppsa_label_col: str = "lable_pre",
    target_col: str = "synthesizability",
    prediction_col: str = "prediction",
    mol_scores_path: Optional[Path] = None,
    mol_score_cols: Optional[List[str]] = None,
    corr_method: str = "spearman",
) -> Dict[str, Any]:
    """Correlate DeepPSA's (inverted) categorical call against RetroTAC, and
    optionally against an external molecular-scores CSV.

    Args:
        deeppsa_preds_path: DeepPSA's per-molecule predictions CSV; must have
            `smiles_col` and `deeppsa_label_col` (values in {0, 1}: 0 = easy,
            1 = hard).
        shared_test_set_path: The shared held-out set built by
            `prepare_test_set.py`; must have `smiles_col` and `target_col`
            (the real-valued ground truth RetroTAC is trained to predict).
        retrotac_preds_path: RetroTAC ensemble predictions on the same set;
            must have `smiles_col` and `prediction_col`.
        output_dir: Directory for `correlation_results.json`, the `.log` file,
            and (only if `mol_scores_path` is given) `correlation_matrix.csv`/
            `.png`. Created if missing.
        smiles_col: SMILES column name, shared by all inputs.
        deeppsa_label_col: DeepPSA's categorical column, 0 = easy, 1 = hard.
        target_col: Real-valued ground-truth column in `shared_test_set_path`.
        prediction_col: RetroTAC's real-valued prediction column in
            `retrotac_preds_path`.
        mol_scores_path: Optional external per-molecule scores CSV. When
            given, triggers the extended SMILES-aligned correlation-matrix
            analysis (see `_molecular_scores_analysis`).
        mol_score_cols: Explicit columns to use from `mol_scores_path`, or
            None to auto-detect every numeric column besides `smiles_col`.
            Ignored when `mol_scores_path` is None.
        corr_method: Correlation method for the `--mol-scores` matrix -- one
            of "pearson"/"spearman"/"kendall". Ignored when `mol_scores_path`
            is None; the two base comparisons below always report all three.

    Returns:
        `{"true_vs_deeppsa": {...}, "retrotac_vs_deeppsa": {...}}`, each an
        inner dict of `pearson_r`/`spearman_rho`/`kendall_tau`/`n_samples`,
        plus `"molecular_scores": {...}` (see `_molecular_scores_analysis`)
        when `mol_scores_path` is given.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = _add_file_handler(output_dir)

    deeppsa = pd.read_csv(deeppsa_preds_path)
    shared_test_set = pd.read_csv(shared_test_set_path)
    retrotac = pd.read_csv(retrotac_preds_path)

    # DeepPSA: 0 = easy, 1 = hard. RetroTAC: 0 = hard, 1 = easy. Flip so both
    # sides agree that higher means easier before correlating against either.
    deeppsa[_DEEPPSA_PRED_COL] = 1 - deeppsa[deeppsa_label_col]

    merged = deeppsa[[smiles_col, _DEEPPSA_PRED_COL]]
    merged = _merge_on_smiles(merged, shared_test_set[[smiles_col, target_col]], "shared_test_set", smiles_col)
    merged = _merge_on_smiles(merged, retrotac[[smiles_col, prediction_col]], "retrotac_preds", smiles_col)

    corr_true = _correlations(merged[target_col].to_numpy(dtype=float),
                              merged[_DEEPPSA_PRED_COL].to_numpy(dtype=float))
    corr_true["n_samples"] = len(merged)
    corr_pred = _correlations(merged[prediction_col].to_numpy(dtype=float),
                              merged[_DEEPPSA_PRED_COL].to_numpy(dtype=float))
    corr_pred["n_samples"] = len(merged)

    results: Dict[str, Any] = {"true_vs_deeppsa": corr_true, "retrotac_vs_deeppsa": corr_pred}

    logger.info("true %s vs inverted DeepPSA label (n=%d): pearson_r=%.4f spearman_rho=%.4f kendall_tau=%.4f",
               target_col, corr_true["n_samples"], corr_true["pearson_r"], corr_true["spearman_rho"], corr_true["kendall_tau"])
    logger.info("RetroTAC %s vs inverted DeepPSA label (n=%d): pearson_r=%.4f spearman_rho=%.4f kendall_tau=%.4f",
               prediction_col, corr_pred["n_samples"], corr_pred["pearson_r"], corr_pred["spearman_rho"], corr_pred["kendall_tau"])

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
    ap.add_argument("--deeppsa-label-col", default="lable_pre",
                    help="DeepPSA's categorical column, 0=easy 1=hard (default: %(default)s)")
    ap.add_argument("--target-col", default="synthesizability",
                    help="real-valued ground-truth column in --shared-test-set (default: %(default)s)")
    ap.add_argument("--prediction-col", default="prediction",
                    help="RetroTAC's real-valued prediction column in --retrotac-preds (default: %(default)s)")
    return ap.parse_args()


def main() -> None:
    """CLI entry point: parse arguments and run the correlation comparison."""
    args = parse_args()
    mol_score_cols = [c.strip() for c in args.mol_score_cols.split(",")] if args.mol_score_cols else None
    compare_deeppsa_retrotac(
        args.deeppsa_preds, args.shared_test_set, args.retrotac_preds, args.output_dir,
        smiles_col=args.smiles_col, deeppsa_label_col=args.deeppsa_label_col,
        target_col=args.target_col, prediction_col=args.prediction_col,
        mol_scores_path=args.mol_scores, mol_score_cols=mol_score_cols, corr_method=args.corr_method,
    )


if __name__ == "__main__":
    main()
