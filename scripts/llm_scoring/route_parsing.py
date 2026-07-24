"""Parse the depth-keyed retrosynthesis route format produced in your CSV.

The `route` column holds a Python-literal dict of the form:

    {1: [['<product> => <prec1>.<prec2>', '<engine_meta>'], ...],
     2: [['<product> => <prec1>.<prec2>', '<engine_meta>'], ...], ...}

Keys are tree depth (1 = the target disconnection). Each depth holds one or
more reactions (branches). `=>` is the RETRO arrow (product on the left, the
precursors it is disconnected into on the right). Precursors are separated by
`.` (standard SMILES component separator, so each precursor is one molecule).

This module flattens that into typed `Reaction`/`ParsedRoute` objects, rebuilds
parent/child links via canonical-SMILES matching, and renders a compact,
token-efficient "compound legend + steps" view for the LLM judge.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Optional

try:
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    _HAS_RDKIT = True
except Exception:  # pragma: no cover
    _HAS_RDKIT = False


def canonical_smiles(smi: str) -> str:
    """Canonicalize with RDKit if available, else strip. Never raises."""
    smi = smi.strip()
    if not _HAS_RDKIT:
        return smi
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(m) if m is not None else smi


def _norm_resolved(x) -> Optional[bool]:
    """Coerce a CSV `resolved` value to a real bool/None.

    pandas may hand us a Python bool, numpy bool, 1/0, NaN, or the strings
    "True"/"False". We need a genuine `is False` test downstream, so normalise."""
    if x is None:
        return None
    if isinstance(x, str):
        s = x.strip().lower()
        if s in ("true", "1", "yes", "t"):
            return True
        if s in ("false", "0", "no", "f", ""):
            return False
        return None
    try:
        import math

        if isinstance(x, float) and math.isnan(x):
            return None
    except Exception:  # pragma: no cover
        pass
    return bool(x)  # bool, numpy.bool_, 1/0


def describe_molecule(smi: str) -> str:
    """Compact, LLM-facing descriptor line for the routeless (unresolved) case.

    When no route is available the judge sees only the SMILES, so a handful of
    RDKit descriptors give it weak structural grounding (size, ring systems,
    stereocentres, macrocyclicity). Degrades to just the SMILES if RDKit is
    missing or the molecule fails to parse. Delete this call if you'd rather the
    judge work from the SMILES alone."""
    if not _HAS_RDKIT:
        return f"SMILES={smi}"
    from rdkit.Chem import Descriptors, rdMolDescriptors

    m = Chem.MolFromSmiles(smi)
    if m is None:
        return f"SMILES={smi} (RDKit could not parse this SMILES)"
    try:
        ri = m.GetRingInfo()
        largest_ring = max((len(r) for r in ri.AtomRings()), default=0)
        stereo = rdMolDescriptors.CalcNumAtomStereoCenters(m)
        unspec = rdMolDescriptors.CalcNumUnspecifiedAtomStereoCenters(m)
        parts = [
            f"formula={rdMolDescriptors.CalcMolFormula(m)}",
            f"MW={Descriptors.MolWt(m):.0f}",
            f"heavy_atoms={m.GetNumHeavyAtoms()}",
            f"rings={ri.NumRings()}",
            f"aromatic_rings={rdMolDescriptors.CalcNumAromaticRings(m)}",
            f"largest_ring={largest_ring}",
            f"macrocycle={'yes' if largest_ring >= 12 else 'no'}",
            f"stereocentres={stereo} (+{unspec} unspecified)",
            f"rotatable_bonds={rdMolDescriptors.CalcNumRotatableBonds(m)}",
            f"HBD={rdMolDescriptors.CalcNumHBD(m)}",
            f"HBA={rdMolDescriptors.CalcNumHBA(m)}",
        ]
        return "; ".join(parts)
    except Exception:  # pragma: no cover
        return f"SMILES={smi}"


@dataclass
class Reaction:
    label: str               # human label used by the LLM: S1, S2, ...
    depth: int
    order_in_depth: int
    product: str             # original SMILES
    precursors: list[str]    # original SMILES
    metadata: str = ""       # per-step note carried from the route source, e.g. "0.0 Unrecognized"
    product_tag: str = ""    # legend tag, e.g. "T", "I1"
    precursor_tags: list[str] = field(default_factory=list)
    # Create the reaction SMILES
    smiles: str = field(init=False)

    def __post_init__(self):
        self.smiles = f"{'.'.join(self.precursors)}>>{self.product}"


@dataclass
class ParsedRoute:
    target: str
    reactions: list[Reaction]
    building_blocks: list[str]
    resolved: Optional[bool] = None
    resolved_depth: Optional[int] = None
    tag_to_smiles: dict[str, str] = field(default_factory=dict)
    unresolved_leaves: list[str] = field(default_factory=list)
    in_stock: bool = False  # resolved with 0 reactions: target itself is a stock item


def parse_route(
    route_cell,
    target: Optional[str] = None,
    building_blocks: Optional[list[str]] = None,
    resolved: Optional[bool] = None,
    resolved_depth: Optional[int] = None,
) -> ParsedRoute:
    """Parse one route cell (str literal or already-eval'd dict) into a ParsedRoute."""
    raw = ast.literal_eval(route_cell) if isinstance(route_cell, str) else route_cell

    reactions: list[Reaction] = []
    n = 0
    for depth in sorted(raw.keys(), key=lambda d: int(d)):
        for j, entry in enumerate(raw[depth]):
            rxn = entry[0]
            meta = entry[1] if len(entry) > 1 else ""
            prod_part, prec_part = rxn.split("=>")
            product = prod_part.strip()
            precursors = [p for p in prec_part.strip().split(".") if p]
            n += 1
            reactions.append(
                Reaction(
                    label=f"S{n}",
                    depth=int(depth),
                    order_in_depth=j,
                    product=product,
                    precursors=precursors,
                    metadata=str(meta).strip(),
                )
            )

    if target is None and reactions:
        min_depth = min(r.depth for r in reactions)
        target = next(r.product for r in reactions if r.depth == min_depth)

    pr = ParsedRoute(
        target=target,
        reactions=reactions,
        building_blocks=list(building_blocks) if building_blocks else [],
        resolved=_norm_resolved(resolved),
        resolved_depth=resolved_depth,
    )
    _assign_tags(pr)
    return pr


def parse_route_row(row: dict) -> ParsedRoute:
    """Parse one pandas row (dict) from the CSV.

    `resolved` is normalised to a real bool/None. If it is False, or the route
    cell is missing/empty, the row is parsed as a routeless molecule (no
    reactions) so the scorer can fall back to route-free synthesizability.
    If `resolved` is True but the route disconnects into nothing (route ==
    "{}"), the target itself is a purchasable stock item; `ParsedRoute.in_stock`
    flags this so the caller can skip LLM scoring entirely and assign a
    perfect score. The `score` column is intentionally ignored (it is not a
    reliable signal)."""
    bbs = row.get("BBs")
    if isinstance(bbs, str):
        try:
            bbs = ast.literal_eval(bbs)
        except (ValueError, SyntaxError):
            bbs = []

    resolved = _norm_resolved(row.get("resolved"))
    route_cell = row.get("route")
    empty_route = (
        route_cell is None
        or isinstance(route_cell, float)  # pandas NaN
        or (isinstance(route_cell, str) and not route_cell.strip())
        or (isinstance(route_cell, dict) and not route_cell)
    )

    # resolved is False -> ignore the route by contract; empty route -> nothing to score.
    if resolved is False or empty_route:
        pr = ParsedRoute(
            target=row.get("SMILES"),
            reactions=[],
            building_blocks=list(bbs) if bbs else [],
            resolved=resolved,
            resolved_depth=row.get("resolved_depth"),
        )
    else:
        pr = parse_route(
            route_cell,
            target=row.get("SMILES"),
            building_blocks=bbs,
            resolved=resolved,
            resolved_depth=row.get("resolved_depth"),
        )

    # A resolved route with 0 reactions (route == "{}") disconnects into
    # nothing: the target IS the stock item, so there is nothing to judge.
    pr.in_stock = pr.resolved is True and not pr.reactions
    return pr


def _assign_tags(pr: ParsedRoute) -> None:
    tag: dict[str, str] = {}
    smiles_of: dict[str, str] = {}

    ct = canonical_smiles(pr.target)
    tag[ct] = "T"
    smiles_of["T"] = pr.target

    for i, bb in enumerate(pr.building_blocks, 1):
        c = canonical_smiles(bb)
        if c not in tag:
            tag[c] = f"B{i}"
            smiles_of[f"B{i}"] = bb

    icount = 0
    for r in pr.reactions:
        c = canonical_smiles(r.product)
        if c not in tag:
            icount += 1
            tag[c] = f"I{icount}"
            smiles_of[f"I{icount}"] = r.product

    ucount = 0
    for r in pr.reactions:
        for p in r.precursors:
            c = canonical_smiles(p)
            if c not in tag:
                ucount += 1
                tag[c] = f"U{ucount}"
                smiles_of[f"U{ucount}"] = p
                pr.unresolved_leaves.append(p)

    for r in pr.reactions:
        r.product_tag = tag[canonical_smiles(r.product)]
        r.precursor_tags = [tag[canonical_smiles(p)] for p in r.precursors]

    pr.tag_to_smiles = smiles_of


@dataclass
class RenderedRoute:
    route_text: str
    meta_text: str


def render_route_for_llm(pr: ParsedRoute) -> RenderedRoute:
    """Compact chemistry-style view: every structure appears once in a legend,
    reactions reference short tags. Keeps huge PROTAC SMILES from repeating."""
    def order_key(t: str):
        rank = {"T": 0, "I": 1, "B": 2, "U": 3}[t[0]]
        num = int(t[1:]) if len(t) > 1 else 0
        return (rank, num)

    kind = {
        "T": "target",
        "I": "intermediate",
        "B": "building block (assume purchasable)",
        "U": "UNRESOLVED leaf (NOT a building block)",
    }

    lines = ["=== COMPOUND LEGEND (each structure listed once) ==="]
    for t in sorted(pr.tag_to_smiles, key=order_key):
        lines.append(f"  [{t}] {kind[t[0]]}: {pr.tag_to_smiles[t]}")

    lines += ["", '=== RETROSYNTHETIC STEPS ("A <= B + C" means A is synthesised FROM B and C) ===']
    for r in sorted(pr.reactions, key=lambda r: (r.depth, r.order_in_depth)):
        # `r.metadata` (e.g. "0.0 Unrecognized") is a stock-availability annotation from the
        # route source, not a feasibility judgement -- despite the wording, it actually means
        # the precursor WAS recognised. It is deliberately not rendered here since the LLM
        # reads "Unrecognized" as a red flag; it is still kept on Reaction for JSONL dumps.
        rhs = " + ".join(f"[{t}]" for t in r.precursor_tags)
        lines.append(f"  {r.label} [depth {r.depth}]: [{r.product_tag}] <= {rhs}")

    meta = []
    # if pr.resolved is not None:
    #     meta.append(f"resolved={pr.resolved}")
    if pr.resolved_depth is not None:
        meta.append(f"resolved_depth={int(pr.resolved_depth)}")
    meta.append(f"n_steps={len(pr.reactions)}")
    meta.append(f"n_building_blocks={len(pr.building_blocks)}")
    if pr.unresolved_leaves:
        meta.append(f"UNRESOLVED_leaves={len(pr.unresolved_leaves)}")

    return RenderedRoute(
        route_text="\n".join(lines),
        meta_text="; ".join(meta),
    )
