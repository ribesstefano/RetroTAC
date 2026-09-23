"""
plot_butina_clustering.py
──────────────────────────
Cluster-quality metrics vs. Butina clustering cutoff (Tanimoto distance),
from the per-cutoff diagnostics swept while choosing the adaptive Butina
cutoff for the held-out split (data/sets/adaptive_cluster_metrics.csv).

Plots min-max-normalized Silhouette and Calinski-Harabasz scores plus an
inverted, min-max-normalized Davies-Bouldin score, alongside their composite
quality score, against cutoff on one axis; the achieved held-out percentage
is plotted on a twin axis, with a vertical rule marking the cutoff actually
chosen.

Usage
-----
    python scripts/dataset/plot_butina_clustering.py \\
        data/sets/adaptive_cluster_metrics.csv

    python scripts/dataset/plot_butina_clustering.py \\
        data/sets/adaptive_cluster_metrics.csv \\
        --output-dir figures/butina_clustering --chosen-cutoff 0.85 \\
        --prefix adaptive_metrics_vs_threshold

Arguments:
    input_csv        CSV with one row per Butina cutoff (columns: cutoff,
                      achieved_pct, silhouette, davies_bouldin,
                      calinski_harabasz, quality_score, ...).
    --output-dir      Directory the figure (pdf/png) is written to (created
                      if missing). Default: figures/butina_clustering.
    --chosen-cutoff   Cutoff to mark with a vertical reference line
                      (default: 0.85).
    --prefix          Output filename stem (default:
                      adaptive_metrics_vs_threshold).
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import pubstyle as ps

ps.apply_style()

#: Raw columns min-max-normalized onto the left "Normalized metric value"
#: axis. `davies_bouldin` is inverted after normalizing, since lower is
#: better for that metric but the other two are "higher is better".
QUALITY_COLUMNS = ("silhouette", "calinski_harabasz", "davies_bouldin")


def normalize_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Min-max normalize the raw cluster-quality columns to [0, 1].

    Rows where `quality_score` is NaN are the degenerate single-cluster
    cutoffs (e.g. cutoff=0.95, where `davies_bouldin` is inf) -- they are
    dropped before computing the min/max so they can't compress the
    normalized range of every other row, and they are never plotted on this
    axis (only `achieved_pct` stays meaningful there).

    Args:
        df: Raw metrics table, one row per cutoff.

    Returns:
        The rows with a valid `quality_score`, with three added columns:
        `silhouette_norm`, `calinski_harabasz_norm` (both plain min-max) and
        `davies_bouldin_norm` (min-max then inverted, so 1.0 means "best").
    """
    valid = df.dropna(subset=["quality_score"]).copy()
    for col in ("silhouette", "calinski_harabasz"):
        lo, hi = valid[col].min(), valid[col].max()
        valid[f"{col}_norm"] = (valid[col] - lo) / (hi - lo)
    lo, hi = valid["davies_bouldin"].min(), valid["davies_bouldin"].max()
    valid["davies_bouldin_norm"] = 1.0 - (valid["davies_bouldin"] - lo) / (hi - lo)
    return valid


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input_csv", type=str, help="CSV with one row per Butina cutoff.")
    p.add_argument(
        "--output-dir", type=str, default="figures/butina_clustering",
        help="Directory the figure (pdf/png) is written to (default: figures/butina_clustering).",
    )
    p.add_argument(
        "--chosen-cutoff", type=float, default=0.85,
        help="Cutoff to mark with a vertical reference line (default: 0.85).",
    )
    p.add_argument(
        "--prefix", type=str, default="adaptive_metrics_vs_threshold",
        help="Output filename stem (default: adaptive_metrics_vs_threshold).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input_csv)

    required = ("cutoff", "achieved_pct", "quality_score", *QUALITY_COLUMNS)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{missing} not found in {args.input_csv}. Available: {df.columns.tolist()}")

    valid = normalize_metrics(df)
    light_blue = ps.PALETTE["light_blue"]
    dark_orange = ps.PALETTE["dark_orange"]
    held_out_color = ps.PALETTE["purple"]
    green = ps.PALETTE["green"]

    # Flatter than the golden-ratio default `ps.set_size()` would give: this
    # figure is placed at 0.5\linewidth in the document, where the plain
    # golden-ratio height reads as taller than a single dual-axis line plot
    # needs. Width still comes from the document's own text width.
    width_in, _ = ps.set_size()
    fig, ax = plt.subplots(figsize=(width_in, width_in * 0.45), layout="constrained")
    ax2 = ax.twinx()

    l_sil, = ax.plot(
        valid["cutoff"], valid["silhouette_norm"], marker="o",
        color=light_blue, label="Silhouette (norm.)",
    )
    l_ch, = ax.plot(
        valid["cutoff"], valid["calinski_harabasz_norm"], marker="o",
        color=green, label="Calinski-Harabasz (norm.)",
    )
    l_db, = ax.plot(
        valid["cutoff"], valid["davies_bouldin_norm"], marker="o",
        color=dark_orange, label="Davies-Bouldin (norm., inverted)",
    )
    l_q, = ax.plot(
        valid["cutoff"], valid["quality_score"], marker="o", color="black",
        linewidth=1.5, label="Quality score (mean)",
    )
    v_line = ax.axvline(
        args.chosen_cutoff, color=ps.STATS["reference_line"], linestyle="--",
        linewidth=0.8, alpha=0.6, label=f"Chosen cutoff ({args.chosen_cutoff:g})",
        zorder=1,
    )

    # Full `df`, not `valid`: the achieved held-out % stays meaningful at the
    # degenerate cutoff (e.g. 0.95) that the quality metrics above exclude.
    l_ho, = ax2.plot(
        df["cutoff"], df["achieved_pct"], marker="s", linestyle=":",
        color=held_out_color, label="Achieved held-out %",
    )

    ax.set_xlabel("Butina cutoff (Tanimoto distance)")
    ax.set_ylabel("Normalized metric value")
    ax.grid(False)
    ax2.set_ylabel("Achieved held-out %", color=held_out_color)
    ax2.tick_params(axis="y", colors=held_out_color)
    ax2.spines["right"].set_visible(True)
    ax2.spines["left"].set_visible(False)
    ax2.grid(False)

    handles = [l_sil, l_ch, l_db, v_line, l_q, l_ho]
    fig.legend(
        handles, [h.get_label() for h in handles],
        loc="outside lower center", ncol=3,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = ps.save_figure(fig, out_dir / args.prefix, formats=("pdf", "png"))
    print(f"Figure saved to {', '.join(str(p) for p in paths)}")


if __name__ == "__main__":
    main()
