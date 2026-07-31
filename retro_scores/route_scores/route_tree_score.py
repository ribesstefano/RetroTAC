"""
retro_scores/route_scores/route_tree_score.py
=============================================
Tiered synthesizability scoring from a single retrosynthesis route tree.

Two things happen here:

1. Gating on (``resolved`` x route-emptiness) into four tiers::

       resolved=True,  route={}        -> purchasable as-is  -> ceiling
       resolved=True,  route=non-empty -> fully solved       -> structural composite
       resolved=False, route=non-empty -> route found, leaves not in stock -> capped
       resolved=False, route={}        -> unsolved           -> floor

   The ``resolved`` flag is checked first: an empty route is ambiguous (it marks
   both the best case, a purchasable molecule, and the worst case, an unsolved
   one), and ``resolved`` is what disambiguates them.

2. Structural metrics on the non-empty trees (number of steps, longest linear
   sequence, coupling fraction, fragment balance),
   combined into a bounded [0, 1] structural score and gated into the tier.

All scoring behaviour comes from a YAML config passed at call time (see
``config/route_scoring.yaml``); no configuration lives in the source, and no
global config object is used -- a :class:`ScoringConfig` is passed explicitly
to every function that needs it.

Usage
-----
Run inside the environment that has RDKit (``mamba activate env-protac-synth``)
for exact heavy-atom counts; the scorer falls back to a SMILES approximation if
RDKit is unavailable.

    uv run python retro_scores/route_scores/route_tree_score.py \\
      --input data/llm_scoring/routes.csv \\
      --config config/route_scoring.yaml \\
      --output data/llm_scoring/routes_scored.csv \\
      --sep '\\t'

The input CSV must contain a route column (a route dict such as
``{1: [['P => R1.R2', label]], ...}``) and a boolean ``resolved`` column. Column
names default to ``route``, ``resolved``, and ``SMILES``; set them in the
``columns:`` block of the config, or override with ``--route-col``,
``--resolved-col``, ``--smiles-col``. The delimiter is auto-detected; pass
``--sep '\\t'`` for tab-separated files. To change the metric weights or the
empty-route anchors, edit the config file -- do not modify the source.

Output: the input CSV plus ``synthesizability``, a ``score_note``, and the
``struct_*`` diagnostic columns. If ``--output`` is omitted, results are written
to ``<input>_scored.csv``.

The module can also be imported: :func:`score_route` scores one route,
:func:`score_dataframe` scores a whole DataFrame.
"""
from __future__ import annotations

import argparse
import ast
import itertools
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

# Optional RDKit: exact heavy-atom counts when available, regex fallback otherwise.
try:
    from rdkit import Chem
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
    _HAVE_RDKIT: bool = True
except ImportError:
    _HAVE_RDKIT = False

import re

# Organic-subset element matcher for the no-RDKit fallback (two-letter first).
_ATOM_RE = re.compile(r"Cl|Br|[BCNOFPSIbcnofps]")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class ScoringConfig:
    """Immutable scoring parameters, loaded from ``config/route_scoring.yaml``."""

    smiles_col: str = "SMILES"
    resolved_col: str = "resolved"
    route_col: str = "route"
    weights: Dict[str, float] = field(
        default_factory=lambda: {
            "lls": 0.50,
            "coupling": 0.30,
            "balance": 0.20,
        }
    )
    neutral_balance: float = 0.5
    # Length term for the LLS: linear decay 1 - (lls-1)/(lls_max-1), anchored so
    # lls=1 -> 1.0 (a one-step route keeps full length credit) and lls=lls_max -> 0.
    lls_max: float = 11.0  # observed max LLS; the length term reaches 0 here
    ceiling: float = 1.0  # empty route + resolved (purchasable): no tree to score
    floor: float = 0.0  # empty route + unresolved (unsolved): no tree to score

    @classmethod
    def from_yaml(cls, path: str) -> "ScoringConfig":
        """Build a config from a YAML file (see ``config/route_scoring.yaml``)."""
        import yaml

        with open(path, encoding="utf-8") as handle:
            raw: Dict[str, Any] = yaml.safe_load(handle) or {}

        cols: Dict[str, str] = raw.get("columns", {})
        transforms: Dict[str, float] = raw.get("transforms", {})
        bands: Dict[str, Any] = raw.get("bands", {})
        return cls(
            smiles_col=cols.get("smiles", cls.smiles_col),
            resolved_col=cols.get("resolved", cls.resolved_col),
            route_col=cols.get("route", cls.route_col),
            weights=raw.get("weights", None) or cls().weights,
            neutral_balance=transforms.get("neutral_balance", cls.neutral_balance),
            lls_max=transforms.get("lls_max", cls.lls_max),
            ceiling=bands.get("ceiling", cls.ceiling),
            floor=bands.get("floor", cls.floor),
        )


# ---------------------------------------------------------------------------
# Molecule helpers
# ---------------------------------------------------------------------------
def heavy_atom_count(smiles: str) -> int:
    """Return the heavy-atom count of a SMILES.

    Uses RDKit for an exact count when available; otherwise approximates by
    counting element symbols in the SMILES string (ignoring hydrogens).
    """
    if _HAVE_RDKIT:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return mol.GetNumHeavyAtoms()
    return max(1, len(_ATOM_RE.findall(smiles)))


# ---------------------------------------------------------------------------
# Parsed route tree
# ---------------------------------------------------------------------------
@dataclass
class RouteTree:
    """A parsed retrosynthesis route as a molecule directed acyclic graph."""

    target: Optional[str]
    reactions: List[Tuple[str, List[str], str, int]]  # (product, reactants, label, depth)
    products: Set[str]
    reactants: Set[str]

    @property
    def leaves(self) -> Set[str]:
        """Building blocks: molecules that appear as a reactant but never a product."""
        return self.reactants - self.products

    @property
    def intermediates(self) -> Set[str]:
        """Non-target products (the internal nodes of the tree)."""
        return self.products - ({self.target} if self.target else set())


def parse_route(route: Dict[Any, Sequence[Sequence[str]]]) -> RouteTree:
    """Turn a ``{depth: [['P => R1.R2', label], ...]}`` dict into a :class:`RouteTree`.

    Reactants are split on ``.``, which also separates salt/mixture components in
    SMILES; this is acceptable for structural counting but may slightly over-count
    the rare salt.
    """
    reactions: List[Tuple[str, List[str], str, int]] = []
    products: Set[str] = set()
    reactants: Set[str] = set()
    for depth, steps in route.items():
        for entry in steps:
            reaction: str = entry[0]
            label: str = entry[1] if len(entry) > 1 else ""
            if "=>" not in reaction:
                continue
            product_part, reactant_part = reaction.split("=>", 1)
            product: str = product_part.strip()
            these_reactants: List[str] = [
                token.strip() for token in reactant_part.strip().split(".") if token.strip()
            ]
            reactions.append((product, these_reactants, label, int(depth)))
            products.add(product)
            reactants.update(these_reactants)

    roots: Set[str] = products - reactants
    target: Optional[str] = next(iter(roots)) if roots else None
    return RouteTree(target, reactions, products, reactants)


# ---------------------------------------------------------------------------
# Structural metrics
# ---------------------------------------------------------------------------
@dataclass
class StructMetrics:
    """Structural descriptors of one route tree plus the composite structural score."""

    n_steps: int = 0
    n_BB: int = 0  # number of building blocks (leaf molecules)  pylint: disable=invalid-name
    max_depth: int = 0
    lls: int = 0  # longest linear sequence (reactions on the critical path)
    coupling_fraction: float = 0.0  # share of steps that join 2+ fragments (convergent couplings)
    avg_branching: float = 0.0  # mean reactants per reaction
    fragment_balance: float = 0.0  # mean min/max heavy-atom ratio over 2+ reactant steps
    structural_score: float = 0.0  # composite of the structural terms, in [0, 1]


def _longest_linear_sequence(tree: RouteTree) -> int:
    """Return the longest root-to-leaf path counted in reactions."""
    children: Dict[str, List[str]] = {}
    for product, reactants, _label, _depth in tree.reactions:
        children.setdefault(product, []).extend(reactants)

    memo: Dict[str, int] = {}

    def depth_of(node: str, seen: frozenset) -> int:
        if node not in children or node in seen:
            return 0
        if node in memo:
            return memo[node]
        result = 1 + max(depth_of(child, seen | {node}) for child in children[node])
        memo[node] = result
        return result

    return depth_of(tree.target, frozenset()) if tree.target else 0


def _step_balance(counts: List[int]) -> Optional[float]:
    """Balance of one coupling step: mean min/max heavy-atom ratio over all
    reactant pairs. For a 2-reactant step this is the single min/max ratio; for
    3+ reactants every fragment contributes (not just the two extremes).
    Returns None if the step's atom counts are unusable.
    """
    pair_ratios = [
        min(a, b) / max(a, b)
        for a, b in itertools.combinations(counts, 2)
        if max(a, b) > 0
    ]
    return sum(pair_ratios) / len(pair_ratios) if pair_ratios else None


def _fragment_balance(tree: RouteTree, neutral: float) -> float:
    """Mean per-step balance over multi-reactant couplings.

    A value near 1 marks a balanced coupling of comparable modules (easier,
    convergent); a value near 0 marks a small fragment added to a large one
    (linear decoration). ``neutral`` is returned when no 2+ reactant step exists.
    """
    step_balances: List[float] = []
    for _product, reactants, _label, _depth in tree.reactions:
        if len(reactants) >= 2:
            counts = [heavy_atom_count(smiles) for smiles in reactants]
            balance = _step_balance(counts)
            if balance is not None:
                step_balances.append(balance)
    return sum(step_balances) / len(step_balances) if step_balances else neutral


def _length_term(lls: int, config: ScoringConfig) -> float:
    """Map the longest linear sequence to a [0, 1] length score.

    Linear decay ``1 - (lls - 1) / (lls_max - 1)`` (clipped at 0). Anchored so
    lls=1 -> 1.0 (a one-step route keeps full length credit) and lls=lls_max
    -> 0. A higher ``lls_max`` softens the penalty on longer routes.
    """
    lls = max(1, lls)
    return max(0.0, 1.0 - (lls - 1) / (config.lls_max - 1))


def compute_metrics(tree: RouteTree, config: ScoringConfig) -> StructMetrics:
    """Compute the structural metrics and composite structural score for one route tree."""
    metrics = StructMetrics()
    if not tree.reactions:
        return metrics

    metrics.n_steps = len(tree.reactions)
    metrics.n_BB = len(tree.leaves)
    metrics.max_depth = max(depth for *_rest, depth in tree.reactions)
    metrics.lls = _longest_linear_sequence(tree)
    couplings = sum(1 for _p, reactants, _l, _d in tree.reactions if len(reactants) >= 2)
    metrics.coupling_fraction = couplings / metrics.n_steps
    metrics.avg_branching = (
        sum(len(reactants) for _p, reactants, _l, _d in tree.reactions) / metrics.n_steps
    )
    metrics.fragment_balance = _fragment_balance(tree, config.neutral_balance)

    # Length term: linear decay anchored at lls=1 -- see _length_term.
    ease_lls = _length_term(metrics.lls, config)
    weights = config.weights
    weighted_sum = (
        weights["lls"] * ease_lls
        + weights["coupling"] * metrics.coupling_fraction
        + weights["balance"] * metrics.fragment_balance
    )
    metrics.structural_score = weighted_sum / sum(weights.values())
    return metrics


# ---------------------------------------------------------------------------
# Gating into the final synthesizability score
# ---------------------------------------------------------------------------
@dataclass
class RouteScore:
    """The tier, final score, and (for non-empty trees) the structural metrics."""

    tier: str
    resolved: bool
    synthesizability: float
    metrics: Optional[StructMetrics] = None
    note: str = ""


def score_route(
    resolved: bool,
    route: Dict[Any, Sequence[Sequence[str]]],
    config: Optional[ScoringConfig] = None,
) -> RouteScore:
    """Score one route by tier, gating on ``resolved`` first.

    The final score is the structural score of the route. The two empty-route
    cases have no tree to score, so they take fixed anchors: ``ceiling`` when
    resolved (purchasable as-is) and ``floor`` when not (unsolved). Any route
    with reactions -- solved or unresolved -- scores at its structural score, so
    an empty route (0) and a partial route (> 0) remain distinguishable.

    An empty route is ambiguous (purchasable vs unsolved); the ``resolved`` flag
    disambiguates the two cases, so it is checked before route-emptiness.
    """
    config = config or ScoringConfig()
    empty: bool = not route

    if resolved and empty:
        return RouteScore("purchasable", True, config.ceiling, note="molecule purchasable as-is")
    if not resolved and empty:
        return RouteScore("unsolved", False, config.floor, note="no route found")

    tree = parse_route(route)
    metrics = compute_metrics(tree, config)
    tier = "solved" if resolved else "unresolved"
    note = "fully solved" if resolved else "partial route (not all leaves in stock)"
    return RouteScore(tier, resolved, metrics.structural_score, metrics, note)


def score_dataframe(
    dataframe: "Any",
    config: Optional[ScoringConfig] = None,
) -> "Any":
    """Add tier, synthesizability, and structural-metric columns to a DataFrame.

    The route column may hold the dict itself or its string repr; empty or
    malformed routes fall through to the unsolved floor.
    """
    config = config or ScoringConfig()
    scores: List[float] = []
    notes: List[str] = []
    # `structural_score` is omitted: it equals `synthesizability` for scored
    # routes and is redundant in the output.
    metric_columns: Dict[str, List[Any]] = {
        name: [] for name in StructMetrics().__dict__ if name != "structural_score"
    }

    for _index, row in dataframe.iterrows():
        route = row[config.route_col]
        if isinstance(route, str):
            try:
                route = ast.literal_eval(route) if route.strip() else {}
            except (ValueError, SyntaxError):
                route = {}
        result = score_route(bool(row[config.resolved_col]), route or {}, config)
        scores.append(result.synthesizability)
        notes.append(result.note)
        metric_values = asdict(result.metrics) if result.metrics else dict.fromkeys(metric_columns)
        for name in metric_columns:
            metric_columns[name].append(metric_values[name])

    dataframe = dataframe.copy()
    dataframe["synthesizability"] = scores
    for name, column in metric_columns.items():
        dataframe[f"struct_{name}"] = column
    dataframe["score_note"] = notes
    return dataframe


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Tiered synthesizability scoring from single retrosynthesis route trees."
    )
    parser.add_argument("--input", "-i", help="input CSV of routes")
    parser.add_argument("--output", "-o", help="output CSV path (default: <input>_scored.csv)")
    parser.add_argument(
        "--config", "-c", help="path to the scoring YAML (default: config/route_scoring.yaml)"
    )
    parser.add_argument("--route-col", help="override the route-dict column name")
    parser.add_argument("--resolved-col", help="override the solvability-flag column name")
    parser.add_argument("--smiles-col", help="override the molecule-id column name")
    parser.add_argument(
        "--sep", default=None, help="CSV delimiter (default: auto-detect; use '\\t' for TSV)"
    )
    return parser.parse_args()


def _load_config(args: argparse.Namespace) -> ScoringConfig:
    """Build a :class:`ScoringConfig` from the YAML and CLI column overrides."""
    config = ScoringConfig.from_yaml(args.config) if args.config else ScoringConfig()
    if args.route_col:
        config.route_col = args.route_col
    if args.resolved_col:
        config.resolved_col = args.resolved_col
    if args.smiles_col:
        config.smiles_col = args.smiles_col
    return config


def main() -> None:
    """Score an input CSV of routes, write the scored CSV, and print a summary."""
    args = parse_args()

    if not args.input:
        raise SystemExit("provide --input CSV")

    import pandas as pd

    config = _load_config(args)

    # sep=None + the python engine sniffs the delimiter; an explicit --sep
    # overrides. Route dicts are comma-heavy, so TSV files must not be read
    # as comma-separated.
    separator: Optional[str] = args.sep.encode().decode("unicode_escape") if args.sep else None
    dataframe = pd.read_csv(args.input, sep=separator, engine="python")
    for column in (config.route_col, config.resolved_col):
        if column not in dataframe.columns:
            raise SystemExit(f"column '{column}' not found. Available: {list(dataframe.columns)}")

    scored = score_dataframe(dataframe, config)

    output_path: str = args.output or f"{Path(args.input).with_suffix('')}_scored.csv"
    scored.to_csv(output_path, index=False)

    print(f"rdkit available: {_HAVE_RDKIT}")
    print(f"\ncategory distribution:\n{scored['score_note'].value_counts().to_string()}")
    print(f"\nsaved -> {output_path}")


if __name__ == "__main__":
    main()