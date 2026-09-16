"""
correlation_analysis.py
────────────────────────
Correlation analysis of per-molecule synthesizability scores in a scored CSV (e.g.
data/retro_scoring/routes_mol_scored.csv, plus a 'synthesizability' column): computes
correlation matrices over the score columns (auto-detected as every numeric column
whose name does not end in "_scaled", so e.g. sa_score is kept and sa_score_scaled is
dropped). The text report computes all three of pearson/spearman/kendall and covers
each separately (descriptive statistics, the correlation matrix as a text table, the
most strongly correlated pairs, and each score's correlation with the target column);
--method only selects which one of the three the heatmap (pdf/svg/png) renders.

Usage
-----
    python scripts/dataset/correlation_analysis.py \\
        data/retro_scoring/routes_mol_scored.csv outputs/analysis/correlation

    python scripts/dataset/correlation_analysis.py \\
        data/retro_scoring/routes_mol_scored.csv outputs/analysis/correlation \\
        --method pearson --high-corr-threshold 0.8

    # write <stem>_mol-only_correlation_{heatmap.pdf,heatmap.svg,heatmap.png,report.txt}
    # instead of the plain <stem>_correlation_* names, so a second run against a
    # different --columns subset doesn't overwrite the first
    python scripts/dataset/correlation_analysis.py \\
        data/retro_scoring/routes_mol_scored.csv outputs/analysis/correlation \\
        --columns sa_score sc_score ra_score --prefix mol-only

Arguments:
    input_csv              CSV to analyse.
    output_dir              Directory the heatmap PNG and text report are written to
                            (created if missing).
    --method                pearson | spearman | kendall -- which method's matrix the
                            heatmap PNG renders (default: spearman). The text report
                            always covers all three regardless of this flag.
    --columns               Explicit list of columns to analyse, overriding
                            auto-detection (default: every numeric column not ending
                            in "_scaled").
    --high-corr-threshold   |r| threshold for the "highly correlated pairs" report
                            section (default: 0.7).
    --target-col            Column to report per-score correlation against in its own
                            report section (default: "synthesizability"; skipped if
                            not among the analysed columns).
    --prefix                Suffix appended to the input CSV's stem when naming output
                            files ("<stem>_<prefix>_correlation_..."), so repeated runs
                            against the same input/output-dir (e.g. different --columns
                            subsets) don't overwrite each other's files (default: none).
"""

import argparse
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

import pubstyle as ps

ps.apply_style()

CORRELATION_METHODS = ["pearson", "spearman", "kendall"]

#: Hardcoded publication display names for the per-molecule synthesizability
#: scores. Any analysed column not listed here (e.g. an ad hoc --columns run)
#: falls back to a generic "snake_case -> Title Case" rendering.
SCORE_DISPLAY_NAMES: dict = {
    "synthesizability": "Synthesizability",
    "sa_score": "SAscore",
    "sc_score": "SCScore",
    "ra_score": "RAscore",
    "syba_score": "SYBA",
    "gasa_pred": "GASA",
    "fs_score": "FSscore",
}

#: Diverging colormap for the correlation heatmap, built from pubstyle's
#: blue/orange contrast pair rather than its green/purple diverging map --
#: the latter is reserved for statistical better/worse outcomes, and a
#: correlation sign carries no such meaning.
CORRELATION_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "corr_diverging", [ps.PALETTE["blue"], "white", ps.PALETTE["dark_orange"]]
)


def display_name(column: str) -> str:
    """Map a score column name to its publication-ready display label.

    Args:
        column: Raw column name (e.g. "sa_score").

    Returns:
        The hardcoded display name if known, else a generic
        "snake_case -> Title Case" fallback.
    """
    return SCORE_DISPLAY_NAMES.get(column, column.replace("_", " ").title())


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input_csv", type=str, help="CSV to analyse.")
    p.add_argument("output_dir", type=str, help="Directory the heatmap PNG and text report are written to.")
    p.add_argument(
        "--method",
        choices=CORRELATION_METHODS,
        default="spearman",
        help="Which method's matrix the heatmap PNG renders (default: spearman). The "
             "text report always covers all three methods regardless of this flag.",
    )
    p.add_argument(
        "--columns",
        nargs="+",
        default=None,
        help="Explicit score columns to analyse (default: auto-detect numeric, non-'_scaled' columns).",
    )
    p.add_argument(
        "--high-corr-threshold",
        type=float,
        default=0.7,
        help="|r| threshold for the highly-correlated-pairs report section (default: 0.7).",
    )
    p.add_argument(
        "--target-col",
        type=str,
        default="synthesizability",
        help="Column to report per-score correlation against (default: synthesizability).",
    )
    p.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="Suffix appended to the input CSV's stem when naming output files, so repeated "
             "runs against the same input/output-dir don't overwrite each other (default: none).",
    )
    return p.parse_args()


def select_score_columns(df: pd.DataFrame, explicit: Optional[List[str]]) -> List[str]:
    """Choose the score columns to analyse.

    Args:
        df: Source DataFrame.
        explicit: User-supplied column names, or None to auto-detect.

    Returns:
        List of column names, in `df`'s column order.

    Raises:
        ValueError: If an explicitly requested column is missing.
    """
    if explicit is not None:
        missing = [c for c in explicit if c not in df.columns]
        if missing:
            raise ValueError(f"--columns {missing} not found. Available: {df.columns.tolist()}")
        return explicit
    return [
        c for c in df.columns
        if not c.endswith("_scaled") and pd.api.types.is_numeric_dtype(df[c])
    ]


def format_descriptive_stats(df: pd.DataFrame, columns: List[str]) -> str:
    """Render per-column count/mean/std/quartiles/missing as an aligned text table.

    Args:
        df: Source DataFrame.
        columns: Columns to summarise.

    Returns:
        Multi-line string table.
    """
    stat_cols = ["count", "mean", "std", "min", "25%", "50%", "75%", "max", "missing", "missing_pct"]
    label_w = max(len(c) for c in columns) + 2
    header = f"  {'column':<{label_w}s}" + "".join(f"{s:>12s}" for s in stat_cols)
    lines = [header]
    for col in columns:
        # `.astype(float)` rather than `df[col].describe()`: bool columns (e.g. "resolved")
        # are `is_numeric_dtype == True` but pandas' own `describe()` silently excludes
        # bool dtype from its default numeric column selection, dropping them from the
        # result instead of raising -- so relying on describe()'s auto-selection here
        # would desync from `columns` for any bool-typed score column.
        s = df[col].astype(float)
        count = int(s.count())
        missing = len(df) - count
        vals = {
            "count": count, "mean": s.mean(), "std": s.std(), "min": s.min(),
            "25%": s.quantile(0.25), "50%": s.quantile(0.5), "75%": s.quantile(0.75), "max": s.max(),
            "missing": missing, "missing_pct": 100 * missing / max(len(df), 1),
        }
        row = f"  {col:<{label_w}s}"
        for stat in stat_cols:
            row += f"{vals[stat]:>12.0f}" if stat in ("count", "missing") else f"{vals[stat]:>12.3f}"
        lines.append(row)
    return "\n".join(lines)


def format_correlation_matrix(corr: pd.DataFrame) -> str:
    """Render a correlation matrix as an aligned text table -- the heatmap, written down.

    Args:
        corr: Square correlation matrix (as returned by `DataFrame.corr`).

    Returns:
        Multi-line string table.
    """
    cols = list(corr.columns)
    label_w = max(len(c) for c in cols) + 2
    cell_w = max(9, max(len(c) for c in cols) + 1)
    header = f"  {'':<{label_w}s}" + "".join(f"{c:>{cell_w}s}" for c in cols)
    lines = [header]
    for row_name in cols:
        row = f"  {row_name:<{label_w}s}"
        for col_name in cols:
            row += f"{corr.loc[row_name, col_name]:>{cell_w}.3f}"
        lines.append(row)
    return "\n".join(lines)


def high_correlation_pairs(corr: pd.DataFrame, threshold: float) -> List[Tuple[str, str, float]]:
    """List column pairs whose |correlation| meets `threshold`, most correlated first.

    Args:
        corr: Square correlation matrix.
        threshold: Minimum |r| to include.

    Returns:
        List of (col_a, col_b, r) triples, each unordered pair appearing once.
    """
    cols = list(corr.columns)
    pairs = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            r = corr.loc[a, b]
            if pd.notna(r) and abs(r) >= threshold:
                pairs.append((a, b, float(r)))
    pairs.sort(key=lambda x: -abs(x[2]))
    return pairs


def target_correlations(corr: pd.DataFrame, target: str) -> Optional[List[Tuple[str, float]]]:
    """Rank every analysed column's correlation with `target`, strongest first.

    Args:
        corr: Square correlation matrix.
        target: Column name to rank correlations against.

    Returns:
        List of (column, r) pairs excluding `target` itself, or None if `target` was
        not among the analysed columns.
    """
    if target not in corr.columns:
        return None
    ranked = [(c, float(corr.loc[target, c])) for c in corr.columns if c != target]
    ranked.sort(key=lambda x: float("inf") if pd.isna(x[1]) else -abs(x[1]))
    return ranked


def plot_heatmap(corr: pd.DataFrame, output_stem: Path, method: str) -> List[Path]:
    """Save a publication-styled correlation heatmap (pdf + svg + png).

    Only the lower triangle (diagonal included) is drawn: a correlation matrix
    is symmetric, so the upper triangle is redundant ink that a reader has to
    visually discount. Cells are colored on pubstyle's blue/orange contrast
    pair rather than its reserved green/purple diverging map, since sign of a
    correlation is not a "better/worse" statistical outcome.

    Args:
        corr: Square correlation matrix.
        output_stem: Output path *without* an extension.
        method: Correlation method name, used in the title/colorbar label.

    Returns:
        The paths written (pdf, svg, png).
    """
    n = len(corr.columns)
    labels = [display_name(c) for c in corr.columns]
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    # width > height: row/column labels plus the colorbar eat into the width
    # only, so a plain (side, side) figsize leaves the square axes shorter
    # than the canvas and constrained layout pads it top and bottom to
    # center it. The 1.3x factor gives that space back to width instead.
    side = min(ps.set_size(fraction=1.0)[0] / 1.3, max(4.2, 0.72 * n + 1.4))
    fig, ax = plt.subplots(figsize=(side * 1.3, side), layout="constrained")

    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap=CORRELATION_CMAP,
        vmin=-1, vmax=1, center=0, square=True, linewidths=1.0, linecolor="white",
        annot_kws={"size": 9}, xticklabels=labels, yticklabels=labels,
        cbar_kws={"label": f"{method.capitalize()} correlation", "shrink": 0.8},
        ax=ax,
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, va="center")
    ax.tick_params(length=0)
    # suptitle (not ax.set_title): centers on the whole figure, including the
    # colorbar, so a long title doesn't overflow past the (narrower) axes.
    fig.suptitle(f"{method.capitalize()} correlation of synthesizability scores")
    return ps.save_figure(fig, output_stem)


class _Tee:
    """Write-through stream that duplicates writes to several underlying streams."""

    def __init__(self, *streams) -> None:
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


@contextmanager
def tee_stdout_to_file(report_path: Path) -> Iterator[None]:
    """Duplicate everything written to stdout into `report_path` as well, for the duration.

    Args:
        report_path: File to create (parent directories included) and write console
            output to. Opened in "w" mode, so an existing file is overwritten.
    """
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_fh = open(report_path, "w")
    original_stdout = sys.stdout
    sys.stdout = _Tee(original_stdout, report_fh)
    try:
        yield
    finally:
        sys.stdout = original_stdout
        report_fh.close()


def resolve_output_paths(input_csv: str, output_dir: str, prefix: Optional[str]) -> Tuple[Path, Path]:
    """Derive the heatmap stem and text report path from the input CSV's stem (+ prefix).

    Args:
        input_csv: Path to the CSV being analysed.
        output_dir: Value of the `output_dir` argument.
        prefix: Value of the `--prefix` argument, or None. When given, appended to the
            stem ("<stem>_<prefix>_correlation_...") so repeated runs against the same
            input/output-dir (e.g. different --columns subsets) don't overwrite each
            other's files.

    Returns:
        (heatmap_stem, report_path), both inside `output_dir`. `heatmap_stem` has no
        extension -- `plot_heatmap` writes it as pdf/svg/png.
    """
    stem = Path(input_csv).stem
    if prefix:
        stem = f"{stem}_{prefix}"
    out_dir = Path(output_dir)
    return out_dir / f"{stem}_correlation_heatmap", out_dir / f"{stem}_correlation_report.txt"


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input_csv)
    columns = select_score_columns(df, args.columns)
    if len(columns) < 2:
        raise ValueError(f"Need at least 2 numeric score columns to correlate, found {columns}.")

    corr_by_method = {m: df[columns].corr(method=m) for m in CORRELATION_METHODS}

    heatmap_stem, report_path = resolve_output_paths(args.input_csv, args.output_dir, args.prefix)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    heatmap_paths = plot_heatmap(corr_by_method[args.method], heatmap_stem, args.method)

    with tee_stdout_to_file(report_path):
        print(f"Loaded {len(df)} rows from {args.input_csv}")
        print(f"Score columns ({len(columns)}): {', '.join(columns)}")
        print(f"Heatmap method: {args.method} (this report covers all of {', '.join(CORRELATION_METHODS)})")

        print("\n" + "=" * 86)
        print("1. DESCRIPTIVE STATISTICS")
        print("=" * 86)
        print(format_descriptive_stats(df, columns))

        letters = "abcdefghijklmnopqrstuvwxyz"

        print("\n" + "=" * 86)
        print("2. CORRELATION MATRICES")
        print("=" * 86)
        for letter, method in zip(letters, CORRELATION_METHODS):
            print(f"\n2{letter}. {method.upper()}")
            print(format_correlation_matrix(corr_by_method[method]))

        print("\n" + "=" * 86)
        print(f"3. HIGHLY CORRELATED PAIRS (|r| >= {args.high_corr_threshold})")
        print("=" * 86)
        for letter, method in zip(letters, CORRELATION_METHODS):
            print(f"\n3{letter}. {method.upper()}")
            pairs = high_correlation_pairs(corr_by_method[method], args.high_corr_threshold)
            if pairs:
                for a, b, r in pairs:
                    print(f"  {a:<20s} <-> {b:<20s}  r = {r:+.3f}")
            else:
                print(f"  none above threshold {args.high_corr_threshold}")

        print("\n" + "=" * 86)
        print(f"4. CORRELATION WITH '{args.target_col}'")
        print("=" * 86)
        for letter, method in zip(letters, CORRELATION_METHODS):
            print(f"\n4{letter}. {method.upper()}")
            ranked = target_correlations(corr_by_method[method], args.target_col)
            if ranked is None:
                print(f"  skipped: '{args.target_col}' not among analysed columns.")
            else:
                for c, r in ranked:
                    print(f"  {c:<20s} r = {'n/a' if pd.isna(r) else f'{r:+.3f}'}")

        print(f"\nHeatmap saved to {', '.join(str(p) for p in heatmap_paths)}")
        print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
