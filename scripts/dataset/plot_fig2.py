"""
plot_fig2.py
────────────
Figure 2: (a) the per-molecule synthesizability score correlation heatmap
(same styling/columns as correlation_analysis.py's corr_mol_scores run)
computed from a scored CSV (e.g. data/retro_scoring/routes_mol_synth_scored.csv),
next to (b) an overlaid density histogram of the target column for the
train_val vs. held-out test splits (data/sets/routes_train_val.csv /
data/sets/routes_test.csv) -- a sanity check that the held-out split didn't
shift the target distribution.

Usage
-----
    python scripts/dataset/plot_fig2.py \\
        data/retro_scoring/routes_mol_synth_scored.csv \\
        data/sets/routes_train_val.csv data/sets/routes_test.csv \\
        figures/

    python scripts/dataset/plot_fig2.py \\
        data/retro_scoring/routes_mol_synth_scored.csv \\
        data/sets/routes_train_val.csv data/sets/routes_test.csv \\
        figures/ --method pearson --prefix fig2_pearson

Arguments:
    mol_scores_csv   CSV with per-molecule synthesizability scores, for panel (a).
    train_val_csv    Development-set (train+val) CSV, for panel (b).
    test_csv         Held-out test-set CSV, for panel (b).
    output_dir       Directory the figure (pdf/svg/png) is written to (created
                     if missing).
    --method         pearson | spearman | kendall for panel (a)'s heatmap
                     (default: spearman).
    --columns        Explicit columns to correlate in panel (a) (default:
                     synthesizability sa_score sc_score ra_score syba_score
                     gasa_pred fs_score).
    --target-col     Column compared between splits in panel (b) (default:
                     synthesizability).
    --plot-density   Plot panel (b) as densities instead of raw counts.
    --prefix         Output filename stem (default: fig2).
"""

import argparse
from pathlib import Path

import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

import pubstyle as ps

ps.apply_style()

#: Default panel-(a) columns -- matches the corr_mol_scores heatmap.
DEFAULT_HEATMAP_COLUMNS = [
    "synthesizability", "sa_score", "sc_score", "ra_score",
    "syba_score", "gasa_pred", "fs_score",
]

#: Hardcoded publication display names, shared with correlation_analysis.py.
SCORE_DISPLAY_NAMES: dict = {
    "synthesizability": "Our score",
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

# ── layout knobs ────────────────────────────────────────────────
# Everything here is in inches and adds up to `FIG_WIDTH_IN`, because the
# figure is written at exactly that size and imported at 100% -- so an inch
# here is an inch on the page, and a point of type is a point of type. Panel
# (a) is square, so it ends up as large as the smaller of its cell's width and
# height: `HEATMAP_SIDE_IN` sets the height side (it drives the figure height)
# and `PANEL_A_CELL_IN` keeps its cell just wide enough to match, so the
# heatmap fills its row rather than floating in it.

#: Figure width, in inches: the document's own ``\linewidth`` (see
#: `pubstyle.TEXT_WIDTH_PT`). Do not raise this to buy the heatmap more room --
#: the figure is imported at ``width=1\linewidth``, so a wider canvas is
#: scaled back down on import and every font in it shrinks to match. Give the
#: panels more room by rebalancing the split below, or more height.
FIG_WIDTH_IN = ps.set_size()[0]

#: Side length, in inches, of panel (a)'s square heatmap. This drives the
#: figure height; panel (b) is then matched to the same height in `main`.
#: At 7 rows this is ~0.30 in per cell, which is what a two-decimal
#: annotation at `pubstyle.ANNOT_FONTSIZE` needs.
HEATMAP_SIDE_IN = 2.10

#: Width of panel (a)'s whole grid cell, in inches: the heatmap square plus
#: the strip its y tick labels and its ``(a)`` need on the left. Panel (b)
#: gets the remainder. Keep this close to what (a) actually occupies -- the
#: figure is no longer saved with a tight bbox, so slack inside a cell stays
#: in the file as white space instead of being cropped away.
PANEL_A_CELL_IN = 2.62

#: Gap between the two panels, as a fraction of the mean axes width
#: (GridSpec's `wspace`). It comes straight out of the panels' own width, so
#: keep it small -- (b)'s own tick and axis labels already separate the two.
PANEL_WSPACE = 0.02

#: Figure height beyond panel (a)'s square axes, in inches: room for the
#: rotated x tick labels, panel (b)'s x label, and the panel letters above.
#: This is the vertical white space -- too large and the panels float in an
#: over-tall canvas, too small and the square heatmap shrinks to make room.
FIG_VERTICAL_PAD_IN = 0.78

#: In-cell annotation size for the heatmap, in points. Shared with every other
#: figure's in-plot annotation text so they match on the page.
HEATMAP_ANNOT_SIZE = ps.ANNOT_FONTSIZE

#: Drop the leading zero from the heatmap's cell values ("-.37" for -0.37).
#: A correlation cannot exceed 1, so the zero carries no information, and at
#: this cell size dropping it is the difference between an annotation that
#: fits inside its cell and one that crowds the gridlines.
HEATMAP_DROP_LEADING_ZERO = True

#: Draw only the lower triangle of the correlation matrix. It is symmetric,
#: so the upper half is redundant -- but blanking it leaves panel (a)'s
#: top-right corner empty, and in a two-panel figure that empty corner sits
#: right next to panel (b) and reads as a large gap between the panels that
#: no width or spacing knob can close. False (the full matrix) is what keeps
#: the two panels looking evenly weighted; set True if you would rather have
#: the tidier triangle and accept the white corner.
MASK_UPPER_TRIANGLE = False

#: Panel-label x positions, per panel, in that axes' own fractional
#: coordinates (0 = axes left edge, negative = outside it). Nudge these to
#: clear each panel's widest y tick label -- (b)'s count ticks are narrower
#: than (a)'s score names, but its y axis label sits outside them.
PANEL_LABEL_X = {"a": -0.25, "b": -0.20}

#: Panel-label y position, shared, in axes fractions (1 = that axes' top
#: edge, so >1 sits above it). The two panels are matched to the same height
#: in `main`, so one shared value puts both labels on the same line.
PANEL_LABEL_Y = 1.13


def display_name(column: str) -> str:
    """Map a score/column name to its publication-ready display label.

    Args:
        column: Raw column name (e.g. "sa_score").

    Returns:
        The hardcoded display name if known, else a generic
        "snake_case -> Title Case" fallback.
    """
    return SCORE_DISPLAY_NAMES.get(column, column.replace("_", " ").title())


#: Panel (b)'s x-axis label, keyed by column name -- separate from
#: `SCORE_DISPLAY_NAMES` because that panel has room to spell a name out in
#: full, unlike a heatmap tick label sharing space with 6 others. Falls back
#: to `display_name()` for any --target-col not listed here.
HISTOGRAM_XLABEL: dict = {
    "synthesizability": "Our synthesizability score",
}


def histogram_xlabel(column: str) -> str:
    """Map a target column to panel (b)'s x-axis label.

    Args:
        column: Raw column name (e.g. "synthesizability").

    Returns:
        The hardcoded label if known, else `display_name(column)`.
    """
    return HISTOGRAM_XLABEL.get(column, display_name(column))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mol_scores_csv", type=str, help="CSV with per-molecule scores, for panel (a).")
    p.add_argument("train_val_csv", type=str, help="Development-set (train+val) CSV, for panel (b).")
    p.add_argument("test_csv", type=str, help="Held-out test-set CSV, for panel (b).")
    p.add_argument("output_dir", type=str, help="Directory the figure (pdf/svg/png) is written to.")
    p.add_argument(
        "--method",
        choices=["pearson", "spearman", "kendall"],
        default="spearman",
        help="Correlation method for panel (a)'s heatmap (default: spearman).",
    )
    p.add_argument(
        "--columns",
        nargs="+",
        default=None,
        help=f"Columns to correlate in panel (a) (default: {' '.join(DEFAULT_HEATMAP_COLUMNS)}).",
    )
    p.add_argument(
        "--target-col",
        type=str,
        default="synthesizability",
        help="Column compared between splits in panel (b) (default: synthesizability).",
    )
    p.add_argument(
        "--plot-density",
        action="store_true",
        help="Plot panel (b) as densities (each split normalized to unit area) "
             "instead of raw counts. Densities make the two splits' shapes "
             "comparable despite the development set being ~7x larger.",
    )
    p.add_argument(
        "--prefix",
        type=str,
        default="fig_2",
        help="Output filename stem (default: fig_2).",
    )
    return p.parse_args()


def draw_heatmap(ax: plt.Axes, corr: pd.DataFrame) -> None:
    """Draw a colorbar-free correlation heatmap onto `ax`.

    Cells are colored on pubstyle's blue/orange contrast pair rather than its
    reserved green/purple diverging map, since the sign of a correlation is
    not a "better/worse" statistical outcome. `MASK_UPPER_TRIANGLE` selects
    between the full matrix and the lower triangle alone.

    Args:
        ax: Target axes.
        corr: Square correlation matrix.
    """
    labels = [display_name(c) for c in corr.columns]
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1) if MASK_UPPER_TRIANGLE else None
    if HEATMAP_DROP_LEADING_ZERO:
        # seaborn's `fmt` is a format spec, which cannot drop a leading zero,
        # so hand it pre-rendered strings and an empty spec instead.
        annot, fmt = corr.map(lambda v: f"{v:.2f}".replace("0.", ".", 1)), ""
    else:
        annot, fmt = True, ".2f"
    sns.heatmap(
        corr, mask=mask, annot=annot, fmt=fmt, cmap=CORRELATION_CMAP,
        vmin=-1, vmax=1, center=0, square=True, linewidths=0.5, linecolor="white",
        annot_kws={"size": HEATMAP_ANNOT_SIZE}, xticklabels=labels, yticklabels=labels,
        cbar=False, ax=ax,
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, va="center")
    ax.tick_params(length=0)


def draw_split_distribution(
    ax: plt.Axes, train_val: pd.Series, test: pd.Series, target_col: str, density: bool
) -> None:
    """Draw overlaid histograms of `target_col` for both splits onto `ax`.

    Args:
        ax: Target axes.
        train_val: Target-column values from the development (train+val) split.
        test: Target-column values from the held-out test split.
        target_col: Name of the column being compared, for the x-axis label.
        density: Normalize each split to unit area (so the two shapes are
            comparable despite their different sizes) instead of plotting raw
            counts, where the larger split dominates the y-axis.
    """
    lo = min(train_val.min(), test.min())
    hi = max(train_val.max(), test.max())
    bins = np.linspace(lo, hi, 31)
    blue, orange = ps.CONTRAST_LIGHT
    heights_tv, _, _ = ax.hist(
        train_val, bins=bins, density=density, color=blue, edgecolor="white",
        linewidth=0.3, alpha=0.65, label="Train/Val", zorder=2,
    )
    heights_te, _, _ = ax.hist(
        test, bins=bins, density=density, color=orange, edgecolor="white",
        linewidth=0.3, alpha=0.65, label="Held-out test", zorder=3,
    )
    ax.set_xlabel(histogram_xlabel(target_col))
    ax.set_ylabel("Density" if density else "Count")
    # Headroom so the legend clears the tallest bar instead of sitting on it.
    ax.set_ylim(0, max(heights_tv.max(), heights_te.max()) * 1.18)
    ax.legend()


def panel_label(ax: plt.Axes, text: str, x: float, y: float) -> None:
    """Draw a bold panel label just outside `ax`'s top-left corner.

    Both coordinates are axes fractions, so the label follows its panel
    wherever the layout engine puts it. `main` matches the two panels to the
    same height, so a shared `y` lands both labels on one line.

    Args:
        ax: Target axes.
        text: Label text, e.g. "(a)".
        x: Position in axes coordinates; see `PANEL_LABEL_X`.
        y: Position in axes coordinates; see `PANEL_LABEL_Y`.
    """
    ax.text(x, y, text, transform=ax.transAxes, fontweight="bold",
            fontsize=ps.PANEL_LABEL_FONTSIZE, va="top", ha="left", clip_on=False)


def main() -> None:
    args = parse_args()

    mol_df = pd.read_csv(args.mol_scores_csv)
    columns = args.columns or DEFAULT_HEATMAP_COLUMNS
    missing = [c for c in columns if c not in mol_df.columns]
    if missing:
        raise ValueError(
            f"--columns {missing} not found in {args.mol_scores_csv}. "
            f"Available: {mol_df.columns.tolist()}"
        )
    corr = mol_df[columns].corr(method=args.method)

    train_val_df = pd.read_csv(args.train_val_csv)
    test_df = pd.read_csv(args.test_csv)
    for path, df in ((args.train_val_csv, train_val_df), (args.test_csv, test_df)):
        if args.target_col not in df.columns:
            raise ValueError(f"--target-col '{args.target_col}' not found in {path}.")

    # The square heatmap sets the row height; everything else follows from it.
    fig = plt.figure(
        figsize=(FIG_WIDTH_IN, HEATMAP_SIDE_IN + FIG_VERTICAL_PAD_IN), layout="constrained"
    )
    # width_ratios in inches: (a)'s cell is sized to just hold its square plus
    # its y labels, and (b) takes everything left over.
    gs = fig.add_gridspec(
        1, 2, width_ratios=(PANEL_A_CELL_IN, FIG_WIDTH_IN - PANEL_A_CELL_IN),
        wspace=PANEL_WSPACE,
    )
    ax_a = fig.add_subplot(gs[0])
    ax_a.set_box_aspect(1)
    ax_b = fig.add_subplot(gs[1])

    draw_heatmap(ax_a, corr)
    draw_split_distribution(
        ax_b, train_val_df[args.target_col], test_df[args.target_col],
        args.target_col, args.plot_density,
    )
    ax_a.grid(False)
    ax_b.grid(False)

    # Match (b)'s plot-box height to (a)'s square so the two panels line up
    # top *and* bottom. The two cells' axes widths aren't proportional to
    # `width_ratios` -- each panel's own tick labels are subtracted from its
    # cell first -- so measure them after one layout pass instead of
    # predicting them. (b) keeps its width here, so this converges in one go.
    fig.canvas.draw()
    ax_b.set_box_aspect(ax_a.get_window_extent().height / ax_b.get_window_extent().width)

    # Both panels have a fixed aspect, so each can be smaller than the cell it
    # was given and the leftover has to go *somewhere*. Anchor them facing each
    # other -- (a) to its cell's top-right, (b) to its top-left -- so any
    # leftover lands on the figure's outer edges rather than *between* the two
    # panels, where it would read as a gap and where shrinking PANEL_WSPACE
    # cannot touch it (the freed width just becomes more slack). The page is
    # saved at exactly `figsize` now, so that edge slack is no longer cropped
    # away either: keep `PANEL_A_CELL_IN` honest and there is little of it.
    # Both stay "north", so the tops still align.
    ax_a.set_anchor("NE")
    ax_b.set_anchor("NW")

    panel_label(ax_a, "(a)", PANEL_LABEL_X["a"], PANEL_LABEL_Y)
    panel_label(ax_b, "(b)", PANEL_LABEL_X["b"], PANEL_LABEL_Y)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = ps.save_figure(fig, out_dir / args.prefix)
    print(f"Figure saved to {', '.join(str(p) for p in paths)}")


if __name__ == "__main__":
    main()
