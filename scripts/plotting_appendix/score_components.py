"""
scripts/plotting_appendix/score_components.py
=============================================
Ablation of the route-score's components, for the appendix section that
argues the adopted two-term score is insensitive to the terms and weights
that were dropped while designing it.

The scorer (`retro_scores/route_scores/route_tree_score.py`) writes every
route-level quantity it derives into `struct_*` columns *before* it collapses
them into `synthesizability`. That makes the composite re-computable from the
scored CSV alone -- no route tree has to be walked again -- so a variant is
just a different weighting/transform of the same columns. This script

  1. re-derives the shipped `synthesizability` column from `struct_*` and
     asserts it matches (so every variant below is on the same footing);
  2. re-scores the dataset under a set of design variants (extra terms,
     different weights, different depth decay, different saturation);
  3. reports, per variant, how far it moves the *ranking* the surrogate is
     trained on -- Spearman/Kendall against the adopted score, mean and max
     absolute shift, and agreement of the 0.7 positive/negative label;
  4. plots the component distributions and the variant score distributions.

Boundary rows (purchasable -> ceiling, unsolved -> floor) carry no route and
therefore take the same anchor under every variant. They are held fixed and
the agreement statistics are reported both on the scorable routes alone and
on the full dataset, since including ~10% of rows that cannot move by
construction inflates every correlation.

Usage
-----
    python scripts/plotting_appendix/score_components.py \\
        --input data/routes/routes_scored_deduped.csv \\
        --config config/route_scoring.yaml

Outputs (see --out-dir / --fig-dir):
    outputs/appendix/score_components_agreement.csv
    outputs/appendix/score_components_terms.csv
    figures/appendix/score_components_terms.{pdf,svg,png}
    figures/appendix/score_components_variants.{pdf,svg,png}
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy import stats

import pubstyle as ps

ps.apply_style()

#: Score band anchors for the two empty-route cases, keyed by the scorer's own
#: `score_note`. These rows have no tree, so no variant can move them.
PURCHASABLE_NOTE = "molecule purchasable as-is"
UNSOLVED_NOTE = "no route found"

#: Threshold the paper uses to turn the continuous score into a binary
#: easy/hard label (config/models_config.yaml: hpo.classification_threshold).
CLF_THRESHOLD = 0.7

#: Two kinds of design change, kept apart because they answer different
#: questions. A *re-parameterisation* keeps the adopted pair of terms and
#: moves one of the free constants that no chemistry fixes (the weight split,
#: the saturation depth, the shape of the decay). A *re-specification* changes
#: which terms enter the score at all.
REPARAMETERISED = "Same two terms, re-parameterised"
RESPECIFIED = "Different term set"

#: Label of the score actually shipped, and the baseline every variant is
#: compared against.
ADOPTED = "Adopted (depth + balance)"


# ---------------------------------------------------------------------------
# Term construction
# ---------------------------------------------------------------------------

def linear_depth(lls: np.ndarray, lls_max: float) -> np.ndarray:
    """Depth term with linear decay, as shipped.

    ``1 - (lls - 1) / (lls_max - 1)``, clipped at 0 and anchored so a
    one-step route keeps full credit and ``lls = lls_max`` scores 0. Note
    this is the anchored form in the code; the manuscript's Eq. 1 writes the
    unanchored ``1 - D/D_max``, which differs by the ``-1`` offsets.

    Args:
        lls: Longest linear sequence per route.
        lls_max: Saturation depth at which the term reaches 0.

    Returns:
        Depth term in [0, 1].
    """
    return np.clip(1.0 - (lls - 1.0) / (lls_max - 1.0), 0.0, 1.0)


def log_depth(lls: np.ndarray, lls_max: float) -> np.ndarray:
    """Depth term with logarithmic decay, the earlier design.

    ``1 - log10(lls) / log10(lls_max)``, clipped at 0. Compresses the
    difference between long routes the way RLScore and RPScore's geometric
    penalty do; the linear form replaced it for interpretability.

    Args:
        lls: Longest linear sequence per route.
        lls_max: Saturation depth at which the term reaches 0.

    Returns:
        Depth term in [0, 1].
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        out = 1.0 - np.log10(np.maximum(lls, 1.0)) / np.log10(lls_max)
    return np.clip(out, 0.0, 1.0)


def convergence_term(lls: np.ndarray, n_steps: np.ndarray) -> np.ndarray:
    """Share of reaction steps that sit off the longest linear sequence.

    0 for a purely linear route (every step is on the LLS), approaching 1 for
    a highly convergent one. A candidate third term computable from the
    existing `struct_*` columns, i.e. with no second route traversal.

    Args:
        lls: Longest linear sequence per route.
        n_steps: Total reaction steps per route.

    Returns:
        Convergence term in [0, 1].
    """
    return np.clip(1.0 - lls / np.maximum(n_steps, 1.0), 0.0, 1.0)


def branching_term(avg_branching: np.ndarray, lo: float = 1.0, hi: float = 3.0) -> np.ndarray:
    """Average branching factor rescaled onto [0, 1].

    `struct_avg_branching` is the mean number of reactants per step, which is
    1 for a purely linear decoration sequence and grows as steps join more
    fragments. Rescaled against its structural bounds (a step has at least
    one reactant; three is the observed maximum) rather than min-max over the
    sample, so the term does not change meaning with the dataset.

    Args:
        avg_branching: Mean reactants per step, per route.
        lo: Value mapped to 0.
        hi: Value mapped to 1.

    Returns:
        Branching term in [0, 1].
    """
    return np.clip((avg_branching - lo) / (hi - lo), 0.0, 1.0)


# ---------------------------------------------------------------------------
# Variants
# ---------------------------------------------------------------------------

#: Each variant is (label, builder). The builder receives the `struct_*`
#: frame and returns the structural-ease score for routes that have a tree.
#: `weights` are normalised by their sum, exactly as the scorer does, so a
#: zero weight removes a term from numerator and denominator alike.
def _weighted(terms: Dict[str, np.ndarray], weights: Dict[str, float]) -> np.ndarray:
    """Normalised weighted mean of named terms.

    Args:
        terms: Term name -> per-route values.
        weights: Term name -> weight; need not sum to 1.

    Returns:
        Composite score per route.
    """
    total = sum(weights.values())
    acc = np.zeros(len(next(iter(terms.values()))), dtype=float)
    for name, weight in weights.items():
        if weight:
            acc = acc + weight * terms[name]
    return acc / total


def build_variants(df: pd.DataFrame, lls_max: float
                   ) -> Dict[str, "tuple[np.ndarray, str]"]:
    """Score every design variant on the routes that have a tree.

    Args:
        df: Frame restricted to scorable routes, with the `struct_*` columns.
        lls_max: Saturation depth from the shipped config.

    Returns:
        Variant label -> (structural-ease score array, design-change family).
    """
    lls = df["struct_lls"].to_numpy(float)
    n_steps = df["struct_n_steps"].to_numpy(float)
    bal = df["struct_fragment_balance"].to_numpy(float)
    # The scorer writes this column as `struct_coupling_fraction`; "coupling"
    # already means a specific reaction class in chemistry, so the term is
    # called "joining" everywhere it is displayed or written out.
    joining = df["struct_coupling_fraction"].to_numpy(float)
    branch = branching_term(df["struct_avg_branching"].to_numpy(float))
    conv = convergence_term(lls, n_steps)

    lin = linear_depth(lls, lls_max)
    log_ = log_depth(lls, lls_max)

    base = {"depth": lin, "balance": bal, "joining": joining,
            "convergence": conv, "branching": branch}

    variants: Dict[str, "tuple[np.ndarray, str]"] = {}
    # The adopted score: equal weights on the linear depth term and balance.
    variants[ADOPTED] = (_weighted(base, {"depth": 0.5, "balance": 0.5}), REPARAMETERISED)
    # Restore the joining term that was dropped, at equal weight ...
    variants["+ joining (equal)"] = (_weighted(
        base, {"depth": 1 / 3, "balance": 1 / 3, "joining": 1 / 3}), RESPECIFIED)
    # ... and at the original hand-set weighting it carried before removal.
    variants["+ joining (0.50/0.30/0.20)"] = (_weighted(
        base, {"depth": 0.50, "joining": 0.30, "balance": 0.20}), RESPECIFIED)
    # Two further candidate terms, both computable from existing struct_* columns.
    variants["+ convergence (equal)"] = (_weighted(
        base, {"depth": 1 / 3, "balance": 1 / 3, "convergence": 1 / 3}), RESPECIFIED)
    variants["+ branching (equal)"] = (_weighted(
        base, {"depth": 1 / 3, "balance": 1 / 3, "branching": 1 / 3}), RESPECIFIED)
    # All five terms at once: the maximally-inclusive score.
    variants["All five terms"] = (_weighted(base, {k: 0.2 for k in base}), RESPECIFIED)
    # Single-term degenerate cases, as the floor on what each term alone buys.
    variants["Depth only"] = (_weighted(base, {"depth": 1.0}), RESPECIFIED)
    variants["Balance only"] = (_weighted(base, {"balance": 1.0}), RESPECIFIED)
    # Re-weighting the two adopted terms away from parity.
    variants["Weights 0.75/0.25"] = (_weighted(
        base, {"depth": 0.75, "balance": 0.25}), REPARAMETERISED)
    variants["Weights 0.25/0.75"] = (_weighted(
        base, {"depth": 0.25, "balance": 0.75}), REPARAMETERISED)
    # Transform / saturation sensitivity.
    variants["Log depth decay"] = (_weighted(
        {**base, "depth": log_}, {"depth": 0.5, "balance": 0.5}), REPARAMETERISED)
    for alt in (7.0, 9.0, 13.0, 15.0):
        variants[f"Saturation $D_\\mathrm{{max}}={alt:.0f}$"] = (_weighted(
            {**base, "depth": linear_depth(lls, alt)},
            {"depth": 0.5, "balance": 0.5}), REPARAMETERISED)
    return variants


# ---------------------------------------------------------------------------
# Agreement statistics
# ---------------------------------------------------------------------------

def variance_shares(terms: Dict[str, np.ndarray],
                    weights: Dict[str, float]) -> Dict[str, float]:
    """Exact additive decomposition of a composite's variance across its terms.

    For ``S = sum_i w_i t_i / sum_i w_i``, bilinearity of covariance gives
    ``Var(S) = sum_i Cov(w_i t_i / sum_j w_j, S)``, so each term's share is
    well defined and the shares sum to 1 even though the terms are
    correlated. A term carrying a large nominal weight but a small share is
    one the composite barely responds to.

    Args:
        terms: Term name -> per-route values.
        weights: Term name -> weight; need not sum to 1.

    Returns:
        Term name -> share of the composite's variance.
    """
    total = sum(weights.values())
    composite = _weighted(terms, weights)
    var = float(np.var(composite, ddof=1))
    shares = {}
    for name, weight in weights.items():
        if not weight:
            continue
        contrib = float(np.cov(weight * terms[name] / total, composite, ddof=1)[0, 1])
        shares[name] = contrib / var
    return shares


def agreement(reference: np.ndarray, candidate: np.ndarray) -> Dict[str, float]:
    """Rank- and value-agreement of a variant score against the adopted one.

    Args:
        reference: Adopted score.
        candidate: Variant score.

    Returns:
        Dict of Spearman rho, Kendall tau, Pearson r, mean/max absolute
        difference, the share of rows whose 0.7 binary label flips, and the
        Jaccard overlap of the top-decile selections.
    """
    delta = np.abs(candidate - reference)
    n_top = max(1, int(round(0.10 * len(reference))))
    top_ref = set(np.argsort(-reference, kind="stable")[:n_top])
    top_cand = set(np.argsort(-candidate, kind="stable")[:n_top])
    return {
        "spearman_rho": float(stats.spearmanr(reference, candidate).statistic),
        "kendall_tau": float(stats.kendalltau(reference, candidate).statistic),
        "pearson_r": float(stats.pearsonr(reference, candidate).statistic),
        "mean_abs_delta": float(delta.mean()),
        "max_abs_delta": float(delta.max()),
        "label_flip_pct": float(
            100.0 * np.mean((reference >= CLF_THRESHOLD) != (candidate >= CLF_THRESHOLD))),
        "top_decile_jaccard": float(
            len(top_ref & top_cand) / len(top_ref | top_cand)),
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_terms(df: pd.DataFrame, lls_max: float, out_stem: Path) -> List[Path]:
    """Distribution of each candidate term over the scorable routes.

    The two adopted terms are drawn in the data-carrying blue, the three
    candidates that were considered and left out in orange. Each panel is
    annotated with the share of routes sitting on the term's modal value --
    the quantity that decides whether a term can rank the dataset at all.

    Args:
        df: Frame restricted to scorable routes.
        lls_max: Saturation depth from the shipped config.
        out_stem: Output path without extension.

    Returns:
        Paths written.
    """
    lls = df["struct_lls"].to_numpy(float)
    n_steps = df["struct_n_steps"].to_numpy(float)
    blue, orange = ps.PALETTE["blue"], ps.PALETTE["dark_orange"]
    panels = [
        (linear_depth(lls, lls_max), "Depth $D$", blue),
        (df["struct_fragment_balance"].to_numpy(float), "Balance $B$", blue),
        (df["struct_coupling_fraction"].to_numpy(float), "Joining", orange),
        (convergence_term(lls, n_steps), "Convergence", orange),
        (branching_term(df["struct_avg_branching"].to_numpy(float)), "Branching", orange),
    ]
    width, _ = ps.set_size()
    fig, axes = plt.subplots(1, 5, figsize=(width, width * 0.26), layout="constrained",
                             sharex=True)
    bins = np.linspace(0, 1, 41)
    for ax, (values, title, color) in zip(axes, panels):
        ax.hist(values, bins=bins, color=color, edgecolor="white",
                linewidth=0.2, zorder=2)
        _, counts = np.unique(np.round(values, 9), return_counts=True)
        ax.set_xlabel(title, fontsize=ps.LABEL_FONTSIZE - 1)
        ax.tick_params(axis="both", which="major", labelsize=ps.ANNOT_FONTSIZE)
        ax.set_xticks([0, 0.5, 1.0])
        ax.text(0.11, 0.95, f"{100 * counts.max() / counts.sum():.0f}%\ntied",
                transform=ax.transAxes, va="top", ha="left",
                fontsize=ps.ANNOT_FONTSIZE, color=ps.darken(color, 0.7))
    axes[0].set_ylabel("Count", fontsize=ps.LABEL_FONTSIZE - 1)
    return ps.save_figure(fig, out_stem)


def plot_variants(reference: np.ndarray, variants: Dict[str, "tuple[np.ndarray, str]"],
                  agreements: pd.DataFrame, highlight: List[str],
                  out_stem: Path) -> List[Path]:
    """Variant score distributions and their rank agreement with the adopted score.

    Panel (b) is coloured by design-change family rather than by how large the
    correlation happens to be, so the figure states the finding instead of
    restating its own x-axis: moving a free constant keeps the ranking, and
    changing which terms enter does not.

    Args:
        reference: Adopted score on scorable routes.
        variants: Variant label -> (score array, family).
        agreements: Per-variant agreement table indexed by label, carrying a
            "family" column.
        highlight: Variant labels drawn as step outlines in panel (a).
        out_stem: Output path without extension.

    Returns:
        Paths written.
    """
    width, _ = ps.set_size()
    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(width, width * 0.42), layout="constrained",
        gridspec_kw={"width_ratios": [1.0, 1.3]})

    bins = np.linspace(0, 1, 41)
    ax_a.hist(reference, bins=bins, color=ps.PALETTE["blue"],
              edgecolor="white", linewidth=0.2, zorder=2, label="Adopted")
    outline_colors = [ps.PALETTE["dark_orange"], ps.PALETTE["light_blue"], "#707070"]
    for label, color in zip(highlight, outline_colors):
        ax_a.hist(variants[label][0], bins=bins, histtype="step",
                  color=color, linewidth=1.1, zorder=4, label=label)
    ax_a.set_xlabel("Structural-ease score")
    ax_a.set_ylabel("Count")
    ax_a.set_xlim(0, 1)
    ax_a.legend(loc="upper left", fontsize=ps.ANNOT_FONTSIZE)

    order = agreements.sort_values("spearman_rho")
    ypos = np.arange(len(order))
    family_color = {REPARAMETERISED: ps.PALETTE["blue"],
                    RESPECIFIED: ps.PALETTE["dark_orange"]}
    colors = [family_color[f] for f in order["family"]]
    ax_b.barh(ypos, order["spearman_rho"], color=colors, edgecolor="white",
              linewidth=0.3, zorder=3)
    ax_b.set_yticks(ypos)
    ax_b.set_yticklabels(order.index, fontsize=ps.ANNOT_FONTSIZE)
    ax_b.set_xlabel("Spearman $\\rho$ with the adopted score")
    ax_b.set_xlim(0, 1.16)
    ax_b.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax_b.grid(axis="x", alpha=0.3)
    ax_b.grid(axis="y", visible=False)
    for y, rho in zip(ypos, order["spearman_rho"]):
        ax_b.text(rho + 0.015, y, f"{rho:.3f}", va="center", ha="left",
                  fontsize=ps.ANNOT_FONTSIZE)
    # Above the axes, not inside it: every horizontal position inside panel
    # (b) is occupied by either a bar or its value label.
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=family_color[f]) for f in family_color]
    ax_b.legend(handles, list(family_color), loc="lower center",
                bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False,
                fontsize=ps.ANNOT_FONTSIZE, handlelength=1.0, columnspacing=1.2)

    for ax, tag, xoff in ((ax_a, "(a)", -0.19), (ax_b, "(b)", -0.52)):
        ax.text(xoff, 1.10, tag, transform=ax.transAxes, fontweight="bold",
                fontsize=ps.PANEL_LABEL_FONTSIZE, va="top", ha="left", clip_on=False)
    return ps.save_figure(fig, out_stem)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed namespace.
    """
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path,
                   default=Path("data/routes/routes_scored_deduped.csv"),
                   help="Scored route CSV carrying the struct_* columns.")
    p.add_argument("--config", type=Path, default=Path("config/route_scoring.yaml"),
                   help="Scorer config, read for lls_max / weights / bands.")
    p.add_argument("--out-dir", type=Path, default=Path("outputs/appendix"),
                   help="Directory for the CSV tables.")
    p.add_argument("--fig-dir", type=Path, default=Path("figures/appendix"),
                   help="Directory for the figures.")
    return p.parse_args()


def main() -> None:
    """Run the ablation and write tables plus figures."""
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    cfg = yaml.safe_load(args.config.read_text())
    lls_max = float(cfg["transforms"]["lls_max"])
    weights = cfg["weights"]
    ceiling = float(cfg["bands"]["ceiling"])
    floor = float(cfg["bands"]["floor"])
    print(f"config: lls_max={lls_max} weights={weights} bands=({floor}, {ceiling})")

    df = pd.read_csv(args.input)
    scorable = df["struct_lls"].notna()
    routes = df.loc[scorable].reset_index(drop=True)
    print(f"{len(df)} rows: {scorable.sum()} with a route, "
          f"{(df['score_note'] == PURCHASABLE_NOTE).sum()} purchasable, "
          f"{(df['score_note'] == UNSOLVED_NOTE).sum()} unsolved")

    variants = build_variants(routes, lls_max)
    adopted = variants[ADOPTED][0]

    # Sanity check: the re-derived adopted score must reproduce the shipped
    # column, or every variant below is measured against the wrong baseline.
    shipped = routes["synthesizability"].to_numpy(float)
    max_dev = float(np.abs(adopted - shipped).max())
    print(f"re-derived vs shipped synthesizability: max |delta| = {max_dev:.2e}")
    if max_dev > 1e-9:
        raise SystemExit("re-derivation does not reproduce the shipped score; "
                         "check config/route_scoring.yaml against the CSV.")

    # Full-dataset view: boundary rows take the same anchor under every
    # variant, so they are pasted back in unchanged.
    anchors = np.where(df["score_note"].to_numpy() == PURCHASABLE_NOTE, ceiling, floor)

    rows = []
    for label, (values, family) in variants.items():
        if label == ADOPTED:
            continue
        stats_routes = agreement(adopted, values)
        full_ref = np.where(scorable.to_numpy(), np.nan, anchors)
        full_cand = full_ref.copy()
        full_ref[scorable.to_numpy()] = adopted
        full_cand[scorable.to_numpy()] = values
        stats_full = agreement(full_ref, full_cand)
        rows.append({
            "variant": label,
            "family": family,
            **{f"{k}": v for k, v in stats_routes.items()},
            **{f"full_{k}": v for k, v in stats_full.items()},
        })
    agreements = pd.DataFrame(rows).set_index("variant")
    agreements.to_csv(args.out_dir / "score_components_agreement.csv")
    print("\n=== agreement with the adopted score (scorable routes only) ===")
    print(agreements[["family", "spearman_rho", "kendall_tau", "pearson_r",
                      "mean_abs_delta", "max_abs_delta", "label_flip_pct",
                      "top_decile_jaccard"]].round(4).to_string())
    print("\n=== same, on the full dataset incl. anchored boundary rows ===")
    print(agreements[["full_spearman_rho", "full_mean_abs_delta",
                      "full_label_flip_pct"]].round(4).to_string())

    # Per-term descriptive statistics, and how much each term alone correlates
    # with the adopted composite.
    lls = routes["struct_lls"].to_numpy(float)
    n_steps = routes["struct_n_steps"].to_numpy(float)
    terms = {
        "Depth (linear)": linear_depth(lls, lls_max),
        "Fragment balance": routes["struct_fragment_balance"].to_numpy(float),
        "Joining fraction": routes["struct_coupling_fraction"].to_numpy(float),
        "Convergence": convergence_term(lls, n_steps),
        "Branching (rescaled)": branching_term(routes["struct_avg_branching"].to_numpy(float)),
    }
    term_rows = []
    for name, values in terms.items():
        _, counts = np.unique(np.round(values, 9), return_counts=True)
        freqs = counts / counts.sum()
        # Effective number of levels: exp(Shannon entropy) of the value
        # distribution. A term that takes many values but puts most of its
        # mass on one of them scores near 1, so this separates "degenerate"
        # from merely "discrete" in a way a distinct-value count does not.
        eff_levels = float(np.exp(-(freqs * np.log(freqs)).sum()))
        # Share of the term already predictable from the two adopted terms:
        # a redundant term cannot add ranking information, whatever weight
        # it is given.
        design = np.column_stack([np.ones(len(values)), terms["Depth (linear)"],
                                  terms["Fragment balance"]])
        beta, *_ = np.linalg.lstsq(design, values, rcond=None)
        resid = values - design @ beta
        redundancy = float(1.0 - resid.var(ddof=1) / values.var(ddof=1))
        term_rows.append({
            "term": name,
            "mean": values.mean(), "std": values.std(ddof=1),
            "min": values.min(), "median": float(np.median(values)), "max": values.max(),
            "iqr": float(np.subtract(*np.percentile(values, [75, 25]))),
            "pct_at_max": float(100.0 * np.mean(values >= 1.0 - 1e-9)),
            "pct_at_mode": float(100.0 * freqs.max()),
            "n_distinct": int(len(counts)),
            "effective_levels": eff_levels,
            "r2_from_adopted_terms": redundancy,
            "spearman_with_adopted": float(stats.spearmanr(values, adopted).statistic),
        })
    term_table = pd.DataFrame(term_rows).set_index("term")
    term_table.to_csv(args.out_dir / "score_components_terms.csv")
    print("\n=== term distributions over the scorable routes ===")
    print(term_table.round(4).to_string())

    # How much of each composite's variance each term actually carries. A
    # term with a nominal weight far above its variance share is one the
    # composite hardly responds to, which is the quantitative form of the
    # claim that adding it changes little.
    base = {"depth": terms["Depth (linear)"], "balance": terms["Fragment balance"],
            "joining": terms["Joining fraction"], "convergence": terms["Convergence"],
            "branching": terms["Branching (rescaled)"]}
    decomps = {
        ADOPTED: {"depth": 0.5, "balance": 0.5},
        "+ joining (equal)": {"depth": 1 / 3, "balance": 1 / 3, "joining": 1 / 3},
        "+ joining (0.50/0.30/0.20)": {"depth": 0.50, "joining": 0.30, "balance": 0.20},
        "+ convergence (equal)": {"depth": 1 / 3, "balance": 1 / 3, "convergence": 1 / 3},
        "+ branching (equal)": {"depth": 1 / 3, "balance": 1 / 3, "branching": 1 / 3},
        "All five terms": {k: 0.2 for k in base},
    }
    decomp_rows = []
    for label, wts in decomps.items():
        shares = variance_shares(base, wts)
        norm = sum(wts.values())
        for name, share in shares.items():
            decomp_rows.append({"variant": label, "term": name,
                                "nominal_weight": wts[name] / norm,
                                "variance_share": share})
    decomp = pd.DataFrame(decomp_rows)
    decomp.to_csv(args.out_dir / "score_components_variance_shares.csv", index=False)
    print("\n=== nominal weight vs share of the composite's variance ===")
    print(decomp.pivot(index="variant", columns="term",
                      values="variance_share").round(4).to_string())

    written = plot_terms(routes, lls_max, args.fig_dir / "score_components_terms")
    # written += plot_variants(
    #     adopted, variants, agreements,
    #     highlight=["+ joining (0.50/0.30/0.20)", "All five terms", "Log depth decay"],
    #     out_stem=args.fig_dir / "score_components_variants")
    print("\nwrote:", *[str(p) for p in written], sep="\n  ")


if __name__ == "__main__":
    main()
