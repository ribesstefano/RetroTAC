"""
sample_route_tree.py
─────────────────────
Picks a resolved retrosynthesis route whose tree "shape" (step count, leaf
count) sits in a chosen size window, saves its target PROTAC SMILES and
building blocks to an output directory, and (by default) renders the whole
route tree as a PNG in the style of a retrosynthesis scheme: target and
building blocks in rounded boxes with 2D structures, intermediates in
between, "⊖" markers on each disconnection edge.

The random seed picks *which* matching route to show, not whether one
exists -- reseed to page through different examples of roughly the same
shape.

Reuses `retro_scores/route_scores/route_tree_score.py`'s `parse_route`/
`RouteTree` for all route-dict parsing (see that module's docstring for the
`{depth: [['P => R1.R2', label]], ...}` format); no parsing logic is
duplicated here.

Usage
-----
    python scripts/dataset/sample_route_tree.py data/routes/routes.csv \\
        --output-dir data/examples/route_trees --seed 0
"""

from __future__ import annotations

import argparse
import ast
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "retro_scores"))

from route_scores.route_tree_score import RouteTree, ScoringConfig, parse_route, score_route  # noqa: E402

Candidate = Tuple[int, RouteTree]


def parse_route_cell(raw: object) -> Dict:
    """Parse one CSV route cell (dict, Python-literal string, or blank/NaN) into a route dict.

    Args:
        raw: A route column value: an already-parsed dict, its string repr,
            or a blank/NaN cell (pandas' representation of an empty CSV cell).

    Returns:
        The route dict, or ``{}`` for a blank cell or one that fails to parse.
    """
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            return ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            return {}
    return {}  # blank/NaN cell, or an empty-string route


def load_resolved_candidates(
    df: pd.DataFrame, route_col: str = "route", resolved_col: str = "resolved"
) -> List[Candidate]:
    """Parse every resolved, non-empty route in *df* into a RouteTree.

    Rows that are unresolved, have an empty/unparseable route (including the
    "purchasable as-is" case: resolved with no route to draw), or whose route
    has no discoverable target are skipped.

    Args:
        df: Routes DataFrame (e.g. loaded from data/routes/routes.csv).
        route_col: Column holding the route dict (or its string repr).
        resolved_col: Boolean-ish column gating candidacy.

    Returns:
        ``(row_index, RouteTree)`` pairs, preserving df's original index.
    """
    resolved_mask = df[resolved_col].astype(str).str.strip().str.lower().eq("true")
    candidates: List[Candidate] = []
    for idx, row in df.loc[resolved_mask].iterrows():
        route = parse_route_cell(row[route_col])
        if not route:
            continue
        tree = parse_route(route)
        if tree.target is None:
            continue
        candidates.append((idx, tree))
    return candidates


def filter_by_shape(
    candidates: Sequence[Candidate],
    min_steps: int,
    max_steps: int,
    min_leaves: int,
    max_leaves: int,
) -> List[Candidate]:
    """Keep only candidates whose step count and leaf count both fall in range (inclusive)."""
    kept: List[Candidate] = []
    for idx, tree in candidates:
        n_steps = len(tree.reactions)
        n_leaves = len(tree.leaves)
        if min_steps <= n_steps <= max_steps and min_leaves <= n_leaves <= max_leaves:
            kept.append((idx, tree))
    return kept


def pick_examples(candidates: Sequence[Candidate], seed: int, n_examples: int) -> List[Candidate]:
    """Seeded sample of up to *n_examples* distinct candidates (no replacement)."""
    k = min(n_examples, len(candidates))
    return random.Random(seed).sample(list(candidates), k)


def build_children_map(tree: RouteTree) -> Dict[str, List[str]]:
    """Map each product SMILES to the reactants its reaction disconnects it into."""
    children_of: Dict[str, List[str]] = {}
    for product, reactants, _label, _depth in tree.reactions:
        children_of.setdefault(product, list(reactants))
    return children_of


def compute_layout(target: str, children_of: Dict[str, List[str]]) -> Dict[str, Tuple[float, float]]:
    """Dendrogram-style (x, y) coordinates for a target-rooted disconnection tree.

    x is BFS depth from the target (cycle-safe). y gives every leaf its own
    sequential row (in depth-first, per-node child order) and centers each
    internal node on the mean y of its direct children, so a single-child
    node inherits its child's row exactly (a straight connector) and a
    forking node sits between its children's rows.

    Args:
        target: Root molecule SMILES.
        children_of: Product -> reactants, as built by :func:`build_children_map`.

    Returns:
        ``{smiles: (x, y)}`` for every node reachable from *target*.
    """
    x_of: Dict[str, int] = {target: 0}
    order: List[str] = [target]
    frontier = [target]
    seen = {target}
    while frontier:
        next_frontier: List[str] = []
        for node in frontier:
            for child in children_of.get(node, []):
                if child in seen:
                    continue
                seen.add(child)
                x_of[child] = x_of[node] + 1
                order.append(child)
                next_frontier.append(child)
        frontier = next_frontier

    y_of: Dict[str, float] = {}
    leaf_rows = [0]  # single-element list so the nested closure can mutate it

    def assign_y(node: str, ancestors: frozenset) -> float:
        kids = [c for c in children_of.get(node, []) if c not in ancestors]
        if not kids:
            y = float(leaf_rows[0])
            leaf_rows[0] += 1
        else:
            y = sum(assign_y(c, ancestors | {node}) for c in kids) / len(kids)
        y_of[node] = y
        return y

    assign_y(target, frozenset())
    return {node: (x_of[node], y_of[node]) for node in order}


def write_route_csv(output_path: Path, target: str, leaves: Sequence[str]) -> Path:
    """Write the target SMILES and its building blocks as a two-column CSV.

    Args:
        output_path: Destination CSV path; parent directories are created.
        target: The whole-PROTAC SMILES (the tree's root).
        leaves: Building-block SMILES, written in the given order.

    Returns:
        *output_path*, for chaining into a summary print.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [{"role": "target", "smiles": target}]
    rows.extend({"role": "building_block", "smiles": leaf} for leaf in leaves)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    return output_path


def trim_whitespace(img: "Image.Image", pad: int = 8) -> "Image.Image":
    """Crop *img* to its non-white content, plus *pad* pixels of margin.

    Args:
        img: Source image (any mode; compared as RGB).
        pad: Margin kept around the detected content, in pixels.

    Returns:
        The cropped image.
    """
    import numpy as np

    arr = np.array(img.convert("RGB"))
    mask = np.any(arr < 250, axis=-1)
    rows = np.where(np.any(mask, axis=1))[0]
    cols = np.where(np.any(mask, axis=0))[0]
    r0, r1 = max(rows[0] - pad, 0), min(rows[-1] + pad, arr.shape[0])
    c0, c1 = max(cols[0] - pad, 0), min(cols[-1] + pad, arr.shape[1])
    return img.crop((c0, r0, c1, r1))


# Generously large working canvas for render_molecule_image: big enough that
# even a full PROTAC target (measured up to ~70 heavy atoms at bond_px=28 ->
# ~1400x360px) doesn't get clipped before trim_whitespace crops it down.
DEFAULT_RENDER_CANVAS: Tuple[int, int] = (2600, 1300)


def render_molecule_image(
    mol: "Chem.Mol", bond_px: int = 28, canvas_size: Tuple[int, int] = DEFAULT_RENDER_CANVAS, pad: int = 8
) -> "Image.Image":
    """Render *mol* at a fixed pixels-per-bond scale, then trim to content.

    Sizing every molecule to the same fixed canvas forces RDKit to auto-fit
    each one to fill it, so a tiny leaf blows up to fill the box while a
    large PROTAC shrinks to fit -- the opposite of "the same molecule drawn
    at the same scale". Drawing every molecule at a fixed bond length onto a
    shared, generously large canvas and then trimming instead keeps one
    consistent chemical scale everywhere; only the final (trimmed) image
    size varies across molecules, which is the point.

    Args:
        mol: A valid (non-None) RDKit molecule.
        bond_px: Target on-screen length of an average bond, in pixels.
        canvas_size: Working canvas size before trimming; must be larger
            than the largest molecule expected at *bond_px*.
        pad: Margin kept around the molecule after trimming.

    Returns:
        The trimmed RGB image.
    """
    import io

    from PIL import Image
    from rdkit.Chem import AllChem
    from rdkit.Chem.Draw import rdMolDraw2D

    AllChem.Compute2DCoords(mol)
    drawer = rdMolDraw2D.MolDraw2DCairo(*canvas_size)
    drawer.drawOptions().fixedBondLength = bond_px
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    img = Image.open(io.BytesIO(drawer.GetDrawingText())).convert("RGB")
    return trim_whitespace(img, pad=pad)


# Box border by role: target (solid cyan), building block / leaf (dashed
# green), intermediate (plain grey) -- every leaf is marked the same way,
# not just one, since structurally they're all equally "a building block".
# Original linewidths target/leaf/intermediate: 2.0/1.6/1.0
_TARGET_STYLE = {"edgecolor": "#2CA8E0", "linewidth": 6.0, "linestyle": "solid", "label": "TARGET"}
_LEAF_STYLE = {"edgecolor": "#4CAF50", "linewidth": 4.8, "linestyle": "dashed", "label": "BUILDING BLOCK"}
_INTERMEDIATE_STYLE = {"edgecolor": "#999999", "linewidth": 3.0, "linestyle": "solid", "label": None}


def _node_style(smiles: str, tree: RouteTree) -> Dict[str, object]:
    if smiles == tree.target:
        return _TARGET_STYLE
    if smiles in tree.leaves:
        return _LEAF_STYLE
    return _INTERMEDIATE_STYLE


def format_target_label(score: float) -> str:
    """Format the target box's label, e.g. ``"TARGET - SCORE: 0.8"``."""
    return f"TARGET - SCORE: {score:.1f}"


def plot_route_tree(
    tree: RouteTree,
    output_path: Path,
    score: float,
    bond_px: int = 28,
    render_canvas: Tuple[int, int] = DEFAULT_RENDER_CANVAS,
    x_spacing: float = 1.25,
    y_spacing: float = 1.3,
    dpi: int = 150,
) -> Path:
    """Render the whole route tree as a retrosynthesis scheme, target to leaves.

    Target and every leaf get an RDKit 2D depiction in a rounded box (solid
    cyan + "TARGET - SCORE: X.Y" for the root, dashed green + "BUILDING
    BLOCK" for every leaf); intermediates get a plain box. Each disconnection
    step is drawn as an elbow connector with a small "⊖" marker, forking when
    the reaction has 2+ reactants. A SMILES that RDKit can't parse falls back
    to a text box instead of failing the whole plot. Every molecule is drawn
    at the same pixels-per-bond scale (see :func:`render_molecule_image`) and
    trimmed to its own content, so a small building block stays small and a
    large target stays large -- boxes vary in size, not in chemical scale.

    Args:
        tree: A parsed route tree (see route_tree_score.parse_route).
        output_path: PNG path to write (a same-named ``.pdf`` is written
            alongside it); parent directories are created.
        score: The route's score, shown on the target label (see
            :func:`format_target_label`).
        bond_px: Target on-screen bond length, in pixels, shared by every
            molecule (see :func:`render_molecule_image`).
        render_canvas: Working canvas passed to :func:`render_molecule_image`.
        x_spacing: Column pitch between depth levels, as a multiple of the
            average of the two neighboring columns' widest molecule (>1
            leaves room for the "-" marker).
        y_spacing: Row pitch between sibling leaves, as a multiple of the
            tallest rendered molecule in the whole tree (>1 leaves room for
            the role label).
        dpi: Resolution the PNG is rasterized at.

    Returns:
        The PNG path (*output_path*); the PDF sibling is written silently.
    """
    from matplotlib import pyplot as plt
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage, TextArea
    import numpy as np
    from rdkit import Chem

    children_of = build_children_map(tree)
    grid = compute_layout(tree.target, children_of)  # raw grid: x in {0,1,2,...}; y in leaf-row units

    # Render every node once, at one shared pixels-per-bond scale (falls back
    # to a small placeholder image for a SMILES RDKit can't parse -- the
    # per-node TextArea used below for that case doesn't need an image at all,
    # so it's simply left out of node_images).
    node_images: Dict[str, "Image.Image"] = {}
    for smiles in grid:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            node_images[smiles] = render_molecule_image(mol, bond_px=bond_px, canvas_size=render_canvas)

    # AnnotationBbox/OffsetImage always place a raw pixel array at exactly
    # (pixels / 72) inches -- 72, not the figure's own dpi -- regardless of
    # what dpi the figure is finally rasterized at (verified empirically:
    # the rendered size in inches is invariant to the figure's dpi setting).
    # So the figure's inch size has to be derived from the actual rendered
    # sizes via that same 72, or every box comes out ~dpi/72x bigger than
    # the space allocated for it. Column width is computed PER COLUMN
    # (grouped by depth), not as a single figure-wide constant: a route with
    # one huge target and several small leaves would otherwise get the
    # target's width applied as the gap between every pair of columns,
    # including ones that hold only small molecules -- exactly the "too much
    # whitespace" failure mode this replaced. Row height stays a single
    # global value: unlike column width, these (mostly chain-like) 2D
    # depictions vary far less in height across a tree, so a per-row max
    # isn't worth the extra bookkeeping.
    POINTS_PER_INCH = 72
    fallback_w, fallback_h = 160, 90  # only reached if every node in a column/tree was unparseable
    columns = sorted({x for x, _y in grid.values()})
    col_width_in: Dict[int, float] = {}
    for col in columns:
        widths = [node_images[s].width for s, (x, _y) in grid.items() if x == col and s in node_images]
        col_width_in[col] = (max(widths) if widths else fallback_w) / POINTS_PER_INCH

    col_x_in: Dict[int, float] = {columns[0]: 0.0}
    for prev_col, cur_col in zip(columns, columns[1:]):
        gap = (col_width_in[prev_col] + col_width_in[cur_col]) / 2 * x_spacing
        col_x_in[cur_col] = col_x_in[prev_col] + gap

    max_img_h = max((img.height for img in node_images.values()), default=fallback_h)
    row_gap_in = (max_img_h / POINTS_PER_INCH) * y_spacing

    positions = {s: (col_x_in[x], y * row_gap_in) for s, (x, y) in grid.items()}
    max_y = max(y for _x, y in grid.values())

    x_lim = (col_x_in[columns[0]] - col_width_in[columns[0]] * 0.6,
              col_x_in[columns[-1]] + col_width_in[columns[-1]] * 0.6)
    y_lim = (-row_gap_in * 0.7, max_y * row_gap_in + row_gap_in * 0.7)
    fig = plt.figure(figsize=(x_lim[1] - x_lim[0], y_lim[1] - y_lim[0]))
    # Full-bleed axes: with no margin to absorb, 1 data unit is exactly 1
    # inch, so the inch-based sizing above actually holds at render time
    # instead of being fudged by subplot margins.
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    ax.set_xlim(*x_lim)
    ax.set_ylim(*y_lim)

    LINE_WIDTH = 3.3 # Original: 1.1
    CIRCLE_SIZE = 50 # Original: 50
    CIRCLE_WIDTH = 3.0 # Original: 1.0
    PLUS_SIZE = 40 # Original: 40

    # Edges first, so the (opaque, white-faced) molecule boxes drawn afterwards
    # cleanly cover the stub end of every connector line.
    for product, reactants in children_of.items():
        if product not in positions:
            continue
        px, py = positions[product]
        child_pts = [positions[r] for r in reactants if r in positions]
        if not child_pts:
            continue
        joint_x = (col_x_in[grid[product][0]] + col_x_in[grid[product][0] + 1]) / 2
        ax.plot([px, joint_x], [py, py], color="#888888", linewidth=LINE_WIDTH, zorder=1)
        child_ys = [cy for _cx, cy in child_pts]
        if len(child_pts) > 1:
            ax.plot([joint_x, joint_x], [min(child_ys), max(child_ys)], color="#888888", linewidth=LINE_WIDTH, zorder=1)
        for cx, cy in child_pts:
            ax.plot([joint_x, cx], [cy, cy], color="#888888", linewidth=LINE_WIDTH, zorder=1)
        ax.plot(
            joint_x, py, marker="o", markersize=CIRCLE_SIZE, markerfacecolor="white",
            markeredgecolor="#555555", markeredgewidth=CIRCLE_WIDTH, zorder=2, clip_on=False,
        )
        ax.text(joint_x, py, "+", ha="center", va="center", fontsize=PLUS_SIZE, color="#555555", zorder=3)

    for smiles, (x, y) in positions.items():
        style = _node_style(smiles, tree)
        bbox_props = dict(
            boxstyle="round,pad=0.35", edgecolor=style["edgecolor"],
            linewidth=style["linewidth"], linestyle=style["linestyle"], facecolor="white",
        )
        content = OffsetImage(np.asarray(node_images[smiles])) if smiles in node_images \
            else TextArea(smiles, textprops=dict(fontsize=6))
        box = AnnotationBbox(
            content, (x, y), frameon=True, pad=0.4, bboxprops=bbox_props,
            zorder=4, annotation_clip=False,
        )
        ax.add_artist(box)
        if style["label"]:
            # Offset from this node's OWN image height (not a tree-wide
            # constant): row_gap_in is sized for the tallest node in the
            # whole tree, so a fixed fraction of it would sit too far below
            # any node shorter than that.
            img_h_in = (node_images[smiles].height if smiles in node_images else fallback_h) / POINTS_PER_INCH
            label_text = format_target_label(score) if smiles == tree.target else style["label"]
            ax.annotate(
                label_text, (x, y + img_h_in / 2 + 0.24), ha="center", va="top",
                fontsize=50, fontweight="bold", color=style["edgecolor"], zorder=4, annotation_clip=False, # Original: fontsize=7.5
            )

    ax.set_xlim(*x_lim)
    ax.set_ylim(*y_lim)
    ax.invert_yaxis()
    ax.axis("off")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # PDF: matplotlib's own tight bbox is exact for vector content, so no
    # extra trimming step is needed (or even possible) here.
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")

    # PNG: bbox_inches="tight" still leaves a little residual margin (it pads
    # to whole pixels and to each artist's own bounding box), so trim once
    # more in pixel space for a genuinely tight crop.
    import io

    from PIL import Image

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    trim_whitespace(Image.open(buffer).convert("RGB"), pad=10).save(output_path)

    return output_path


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_csv", type=str, help="Routes CSV to sample from (e.g. data/routes/routes.csv).")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to write example(s) into.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed selecting which matching route(s) to pick.")
    parser.add_argument("--n-examples", type=int, default=1, help="Number of distinct routes to sample.")
    parser.add_argument("--min-steps", type=int, default=2, help="Minimum number of disconnection steps.")
    parser.add_argument("--max-steps", type=int, default=4, help="Maximum number of disconnection steps.")
    parser.add_argument("--min-leaves", type=int, default=2, help="Minimum number of building blocks (leaves).")
    parser.add_argument("--max-leaves", type=int, default=4, help="Maximum number of building blocks (leaves).")
    parser.add_argument(
        "--plot", action=argparse.BooleanOptionalAction, default=True,
        help="Render each picked route as a tree PNG (default: on; needs matplotlib).",
    )
    parser.add_argument("--route-col", type=str, default="route", help="Route-dict column name.")
    parser.add_argument("--resolved-col", type=str, default="resolved", help="Solvability-flag column name.")
    parser.add_argument(
        "--scoring-config", type=str, default="config/route_scoring.yaml",
        help="YAML config for the structural score shown on the target label (see route_tree_score.py).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Pick matching route(s) from --input_csv and write SMILES + tree PNG to --output-dir."""
    args = parse_args(argv)

    df = pd.read_csv(args.input_csv)
    for column in (args.route_col, args.resolved_col):
        if column not in df.columns:
            raise SystemExit(f"column '{column}' not found. Available: {list(df.columns)}")

    candidates = load_resolved_candidates(df, route_col=args.route_col, resolved_col=args.resolved_col)
    shaped = filter_by_shape(candidates, args.min_steps, args.max_steps, args.min_leaves, args.max_leaves)
    print(
        f"{len(candidates)} resolved routes with a tree; {len(shaped)} fall inside "
        f"{args.min_steps}-{args.max_steps} steps / {args.min_leaves}-{args.max_leaves} leaves."
    )
    if not shaped:
        raise SystemExit(
            "No route matches that window -- widen --min-steps/--max-steps/--min-leaves/--max-leaves."
        )

    picked = pick_examples(shaped, args.seed, args.n_examples)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Loaded lazily, and only used below when --plot is set: scoring needs
    # RDKit heavy-atom counts, so it stays out of the (pure Python, fast)
    # selection pass above and off the --no-plot path entirely.
    scoring_config = ScoringConfig.from_yaml(args.scoring_config) if args.plot else None

    for rank, (row_index, tree) in enumerate(picked):
        stem = f"example_{rank:02d}_row{row_index}"
        leaves = sorted(tree.leaves)
        csv_path = write_route_csv(output_dir / f"{stem}.csv", tree.target, leaves)
        print(f"[{stem}] {len(tree.reactions)} steps, {len(leaves)} building blocks -> {csv_path}")
        if args.plot:
            route = parse_route_cell(df.loc[row_index, args.route_col])
            score = score_route(True, route, scoring_config).synthesizability
            png_path = plot_route_tree(tree, output_dir / f"{stem}_tree.png", score=score)
            print(f"[{stem}] tree plot -> {png_path}")


if __name__ == "__main__":
    main()
