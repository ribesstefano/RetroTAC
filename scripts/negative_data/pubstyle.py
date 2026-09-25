"""Publication figure style: palette, rcParams, and sizing helpers.

Drop this file into a project and call :func:`apply_style` once, before
creating any figure. It has no dependencies beyond matplotlib.

The palette is a two-way contrast pair (blue / orange) for data series, plus
a reserved set (green / purple / grey) used *only* to encode statistical
outcomes. See SKILL.md for the rules and recipes.md for per-figure code.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from cycler import cycler

__all__ = [
    "PALETTE",
    "STATS",
    "CONTRAST",
    "CONTRAST_LIGHT",
    "DIVERGING_CMAP",
    "DIVERGING_CMAP_R",
    "RC_PARAMS",
    "TEXT_WIDTH_PT",
    "TITLE_FONTSIZE",
    "LABEL_FONTSIZE",
    "TICK_FONTSIZE",
    "LEGEND_FONTSIZE",
    "ANNOT_FONTSIZE",
    "PANEL_LABEL_FONTSIZE",
    "apply_style",
    "set_size",
    "darken",
    "save_figure",
    "write_mplstyle",
]

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------

#: Named colors. ``blue``/``orange`` (and their saturated variants) carry data;
#: ``green``/``purple`` are reserved for statistical outcomes.
PALETTE: dict[str, str] = {
    "blue": "#4B9ECE",
    "orange": "#FFAA6E",
    "light_blue": "#50B1D8",
    "dark_orange": "#FF8428",
    "green": "#9DCE9C",
    "purple": "#C8ABDA",
}

#: Reserved roles. Never use these for a data series -- a reader who learns
#: that green means "wins" in one figure must not meet green as a cell line
#: in the next one.
STATS: dict[str, str] = {
    "better": PALETTE["green"],
    "worse": PALETTE["purple"],
    "nonsignificant": "#CCCCCC",
    "reference_line": "#808080",
}

#: The two-way contrast pair for solid fills (bars, boxes, histograms).
CONTRAST: tuple[str, str] = (PALETTE["blue"], PALETTE["dark_orange"])

#: The same pair for translucent or rasterized marks (scatter, KDE fills),
#: where the softer orange reads better at low alpha.
CONTRAST_LIGHT: tuple[str, str] = (PALETTE["blue"], PALETTE["orange"])

#: Diverging map for effect sizes: purple = worse, white = no effect,
#: green = better. Registered under the name ``"pub_diverging"``.
DIVERGING_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "pub_diverging", [PALETTE["purple"], "white", PALETTE["green"]]
)
DIVERGING_CMAP_R = DIVERGING_CMAP.reversed()

for _cmap in (DIVERGING_CMAP, DIVERGING_CMAP_R):
    try:
        mpl.colormaps.register(_cmap)
    except ValueError:  # already registered on module reload
        pass

# ---------------------------------------------------------------------------
# Typography and canvas
# ---------------------------------------------------------------------------

#: Type sizes in points, for a figure *authored at the document's text width
#: and imported at 100%* (see :func:`set_size`). At that scale these are the
#: sizes the reader actually sees, so they are picked against the document's
#: body text -- 10 pt in ICLR: titles and axis labels sit just under it, ticks
#: and legends a step below, annotation text smaller again. Scaling a figure in
#: ``\includegraphics`` is what breaks this correspondence, and what makes two
#: neighbouring figures disagree about how big "a tick label" is.
TITLE_FONTSIZE = 8
LABEL_FONTSIZE = 8
TICK_FONTSIZE = 7
LEGEND_FONTSIZE = 7

#: In-plot text that annotates rather than labels: metrics boxes, bar value
#: labels, heatmap cell values. The smallest size that still prints legibly.
ANNOT_FONTSIZE = 6

#: Bold ``(a)``/``(b)`` panel letters. The house rule is 1.5x the axis-label
#: size; at true scale that would out-shout the 10 pt body text, so 1.25x.
PANEL_LABEL_FONTSIZE = 10

#: LaTeX ``\the\linewidth`` in TeX points (1/72.27 in). 397.48 pt = 5.5 true
#: inches, the single-column ``\textwidth`` set by ``iclr2026_conference.sty``
#: (and by NeurIPS). Put ``\the\linewidth`` in your own document, read the
#: value from the log, and substitute it here: every figure derives its width
#: from this one number, which is what keeps their type sizes in agreement.
TEXT_WIDTH_PT = 397.48

DPI = 300

RC_PARAMS: dict[str, Any] = {
    # Type
    "font.family": "sans-serif",
    "font.size": TICK_FONTSIZE,
    "axes.titlesize": TITLE_FONTSIZE,
    "axes.labelsize": LABEL_FONTSIZE,
    "xtick.labelsize": TICK_FONTSIZE,
    "ytick.labelsize": TICK_FONTSIZE,
    "legend.fontsize": LEGEND_FONTSIZE,
    # Frame: no box, horizontal rules only, behind the data
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "axes.axisbelow": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "-",
    "grid.linewidth": 0.4,
    # Stroke weights, in points, and therefore on the same true scale as the
    # type above: a figure imported at 100% shows these widths as printed. The
    # matplotlib defaults are drawn for a figure that will be shrunk on import
    # and look coarse once it is not.
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.2,
    "lines.markersize": 4,
    "patch.linewidth": 0.5,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.major.pad": 2,
    "ytick.major.pad": 2,
    "axes.labelpad": 2.5,
    "axes.titlepad": 3,
    # Legend: square, hairline, translucent -- and with furniture scaled to a
    # 7 pt entry, or the frame takes up more of the panel than the entries do.
    "legend.frameon": True,
    "legend.fancybox": False,
    "legend.framealpha": 0.5,
    "legend.edgecolor": "#CCCCCC",
    "legend.facecolor": "white",
    "legend.handlelength": 1.4,
    "legend.handletextpad": 0.5,
    "legend.borderpad": 0.35,
    "legend.borderaxespad": 0.4,
    "legend.labelspacing": 0.3,
    # Default series order, so bare df.plot() / sns.barplot() land on-palette
    "axes.prop_cycle": cycler(
        color=[
            PALETTE["blue"],
            PALETTE["dark_orange"],
            PALETTE["light_blue"],
            PALETTE["orange"],
            PALETTE["green"],
            PALETTE["purple"],
        ]
    ),
    # Output
    "figure.dpi": 100,
    "savefig.dpi": DPI,
    # NOT "tight": a tight bbox re-crops the page to whatever ink the figure
    # happens to contain, so the saved width is no longer the width that was
    # asked for, and ``\includegraphics[width=\linewidth]`` then scales it by
    # a factor nobody chose -- differently for every figure. Keeping the page
    # exactly `figsize` is what makes 8 pt in the script mean 8 pt on the page.
    # The cost is that the layout engine has to fit the decorations itself:
    # always use `layout="constrained"`, and check nothing is clipped.
    "savefig.bbox": None,
    "savefig.pad_inches": 0.0,
    # Embed real TrueType in vector output so journals (and Illustrator) can
    # select and reflow the text instead of receiving outlines.
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
}


def apply_style(font_family: str | None = None, font_scale: float = 1.0) -> None:
    """Install the publication style into the global matplotlib rcParams.

    Call once at import time, and *after* any ``seaborn.set_theme`` or
    ``seaborn.set_style`` call -- seaborn overwrites rcParams wholesale and
    will otherwise undo this.

    Args:
        font_family: Override the font family (e.g. ``"Nimbus Sans"`` to match
            a LaTeX document). Defaults to the matplotlib sans-serif stack.
        font_scale: Multiply every font size by this factor. Use values above
            1.0 for figures that will be printed small, or for slides.
    """
    params = dict(RC_PARAMS)
    if font_family is not None:
        params["font.family"] = font_family
    if font_scale != 1.0:
        for key in (
            "font.size",
            "axes.titlesize",
            "axes.labelsize",
            "xtick.labelsize",
            "ytick.labelsize",
            "legend.fontsize",
        ):
            params[key] = params[key] * font_scale
    plt.rcParams.update(params)


def set_size(
    width_pt: float = TEXT_WIDTH_PT,
    fraction: float = 1.0,
    subplots: tuple[int, int] = (1, 1),
) -> tuple[float, float]:
    """Compute a figure size in inches that needs no scaling in LaTeX.

    Sizing the figure to the text width and importing it at 100% keeps the
    figure's font size equal to the document's. Scaling a figure in
    ``\\includegraphics`` is what makes axis labels shrink out of step
    between one figure and the next.

    Args:
        width_pt: Target width in TeX points, i.e. the value of
            ``\\the\\textwidth`` (or ``\\the\\columnwidth``) in your document.
        fraction: Fraction of that width the figure should occupy.
        subplots: ``(nrows, ncols)``; the height is scaled by their ratio so a
            grid keeps roughly golden-ratio cells.

    Returns:
        ``(width_in, height_in)``, ready to pass as ``figsize``.
    """
    fig_width_in = width_pt * fraction / 72.27
    golden_ratio = (5 ** 0.5 - 1) / 2
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])
    return fig_width_in, fig_height_in


def darken(color: str, factor: float = 0.80) -> tuple[float, float, float]:
    """Scale a color's RGB components toward black.

    Used to derive the KDE / trend line color from the fill color it sits on,
    so the line reads as the same hue rather than a second series.

    Args:
        color: Any matplotlib color specification.
        factor: Multiplier applied to each RGB channel; lower is darker.

    Returns:
        An ``(r, g, b)`` tuple.
    """
    r, g, b = mcolors.to_rgb(color)
    return (r * factor, g * factor, b * factor)


def save_figure(
    fig: plt.Figure,
    path_stem: str | Path,
    formats: Iterable[str] = ("pdf", "svg", "png"),
    close: bool = True,
) -> list[Path]:
    r"""Write a figure once per format, creating the parent directory.

    Always keep a vector format: ``pdf`` for LaTeX, ``svg`` for the last-mile
    edits reviewers ask for. ``png`` is for previews and slide decks.

    The page comes out at exactly ``fig.get_size_inches()`` -- see
    ``savefig.bbox`` in :data:`RC_PARAMS` for why that matters. So a figure
    built with ``figsize=set_size()`` and imported with
    ``\includegraphics[width=\linewidth]`` is reproduced at 1:1, and the point
    sizes in :data:`RC_PARAMS` are the point sizes on the page.

    Args:
        fig: The figure to write.
        path_stem: Output path *without* an extension.
        formats: Extensions to write.
        close: Close the figure afterwards, so a loop over many figures does
            not exhaust matplotlib's figure limit.

    Returns:
        The paths written.
    """
    stem = Path(path_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for fmt in formats:
        out = stem.with_suffix(f".{fmt}")
        fig.savefig(out, format=fmt)
        written.append(out)
    if close:
        plt.close(fig)
    return written


def _strip_hash(value: Any) -> Any:
    """Drop the leading ``#`` from a hex color for ``.mplstyle`` output.

    ``#`` opens a comment in a style file, so ``legend.edgecolor: #CCCCCC``
    parses as an empty value and is silently discarded. matplotlibrc's own
    convention is bare hex.
    """
    if isinstance(value, str) and value.startswith("#"):
        return value[1:]
    return value


def write_mplstyle(path: str | Path = "pubstyle.mplstyle") -> Path:
    """Generate a declarative ``.mplstyle`` from :data:`RC_PARAMS`.

    The file carries the rcParams only -- type, grid, spines, legend, output
    settings and the color cycle. It cannot carry :data:`PALETTE`'s semantics,
    the diverging colormap, or the per-figure recipes, so a project using the
    ``.mplstyle`` alone gets the typography but not the look. Prefer importing
    this module; use the generated file for non-Python consumers or projects
    that must not take a local import.

    Args:
        path: Destination file.

    Returns:
        The path written.
    """
    out = Path(path)
    lines = ["# Generated by pubstyle.py -- edit pubstyle.py, not this file.", ""]
    for key, value in RC_PARAMS.items():
        if key == "axes.prop_cycle":
            colors = ", ".join(f"'{_strip_hash(c)}'" for c in value.by_key()["color"])
            lines.append(f"axes.prop_cycle: cycler('color', [{colors}])")
        else:
            lines.append(f"{key}: {_strip_hash(value)}")
    out.write_text("\n".join(lines) + "\n")
    return out


def _main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--emit-mplstyle",
        nargs="?",
        const="pubstyle.mplstyle",
        metavar="PATH",
        help="Write the rcParams subset as a .mplstyle file and exit.",
    )
    args = parser.parse_args(argv)
    if args.emit_mplstyle:
        print(f"Wrote {write_mplstyle(args.emit_mplstyle)}")
    else:
        parser.print_help()


if __name__ == "__main__":
    _main()
