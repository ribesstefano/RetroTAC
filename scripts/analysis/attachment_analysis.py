"""
Stage 1, step 3 — characterize the chemical environment at each attachment point.

For every component, walks up to ``n_hops`` bonds away from the attachment (dummy)
atom, encodes the local environment as canonical SMARTS, and classifies it into a
coarse category (aliphatic C/N, aromatic C, oxygen, carbonyl C, …). The resulting
distribution justifies the small cap set used in step 4.

Requires the PROTAC-Splitter environment (``canonize_smarts``).

I/O
---
    in :  data/processed/protac_smiles_master_std.csv
    out:  data/processed/analysis/attachment_side_classified.csv
          figures/attachment_env.png
"""

import argparse
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from rdkit import Chem

from protac_splitter.chemoinformatics import canonize_smarts

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ── SMARTS → environment label ────────────────────────────────────────────────

SMARTS_TO_ENVIRONMENT: Dict[str, str] = {
    "[#6]-[#7](-[#6])-[#0:1]": "Aliphatic N",
    "[#6]-[#7](-[#6])-[#0:2]": "Aliphatic N",
    "[H]-[#7](-[#6])-[#0:1]":  "Aliphatic N",
    "[H]-[#7](-[#6])-[#0:2]":  "Aliphatic N",

    "[#6]-[#6](=[#8])-[#0:1]": "Carbonyl C",
    "[#6]-[#6](=[#8])-[#0:2]": "Carbonyl C",

    "[#6]:[#6](:[#6])-[#0:1]": "Aromatic C",
    "[#6]:[#6](:[#6])-[#0:2]": "Aromatic C",
    "[#6]:[#6](:[#7])-[#0:1]": "Aromatic C",
    "[#6]:[#6](:[#7])-[#0:2]": "Aromatic C",
    "[#7]:[#6](:[#7])-[#0:1]": "Aromatic C",

    "[H]-[#6](-[#6])(-[#6])-[#0:1]": "Aliphatic C",
    "[H]-[#6](-[#6])(-[#6])-[#0:2]": "Aliphatic C",
    "[H]-[#6](-[H])(-[#6])-[#0:1]":  "Aliphatic C",
    "[H]-[#6](-[H])(-[#6])-[#0:2]":  "Aliphatic C",

    "[#6]-[#8]-[#0:1]": "Oxygen",
    "[#6]-[#8]-[#0:2]": "Oxygen",
}

FALLBACK_ENVIRONMENT = "others"

# linker has two attachment points — one toward the warhead ([*:1]) and one
# toward the E3 ligase ([*:2]).
COMPONENT_MAP: Dict[str, List[Tuple[str, int]]] = {
    "warhead": [("warhead",    1)],
    "e3":      [("e3",         2)],
    "linker":  [("linker_cap1", 1), ("linker_cap2", 2)],
}


# ── Chemistry helpers ─────────────────────────────────────────────────────────

def safe_mol(smiles: str) -> Tuple[Optional[Chem.Mol], str]:
    """Parse SMILES, returning (mol, "") on success or (None, reason) on failure.

    Args:
        smiles: SMILES string to parse.

    Returns:
        Tuple of (RDKit Mol or None, error reason string; empty on success).
    """
    if pd.isna(smiles) or not str(smiles).strip():
        return None, "empty smiles"
    mol = Chem.MolFromSmiles(smiles)
    return (mol, "") if mol is not None else (None, "invalid smiles")


def attachment_environment(
    substruct: Chem.Mol,
    attachment_id: Optional[int] = None,
    n_hops: int = 2,
    add_hs: bool = True,
) -> Optional[str]:
    """Return the canonical SMARTS of the n-hop neighborhood around an attachment point.

    Locates the dummy atom ([*:attachment_id]) in ``substruct``, collects all
    atoms within ``n_hops`` bonds via BFS, and encodes them as canonical SMARTS.

    Args:
        substruct:     Component molecule containing one or more dummy atoms ([*:n]).
        attachment_id: Map number of the target dummy atom (1 or 2). If None,
                       the first dummy atom found is used.
        n_hops:        Bond-distance radius around the dummy atom to include.
        add_hs:        Whether to add explicit hydrogens before analysis.

    Returns:
        Canonical SMARTS string, or None if the attachment point is absent or
        SMARTS generation fails.
    """
    if add_hs:
        substruct = Chem.AddHs(substruct)

    dummy_idx = None
    for atom in substruct.GetAtoms():
        if atom.GetAtomicNum() == 0:
            if attachment_id is None or atom.GetAtomMapNum() == attachment_id:
                dummy_idx = atom.GetIdx()
                break

    if dummy_idx is None:
        return None

    # BFS
    visited = {dummy_idx}
    frontier = deque([(dummy_idx, 0)])
    while frontier:
        idx, depth = frontier.popleft()
        if depth >= n_hops:
            continue
        for nb in substruct.GetAtomWithIdx(idx).GetNeighbors():
            nidx = nb.GetIdx()
            if nidx not in visited:
                visited.add(nidx)
                frontier.append((nidx, depth + 1))

    smarts = Chem.MolFragmentToSmarts(substruct, list(visited))
    return canonize_smarts(smarts) if smarts else None


# ── Analysis table ────────────────────────────────────────────────────────────

def build_attachment_analysis(
    df: pd.DataFrame,
    component_col: str = "components",
    smiles_col: str = "component_smiles_with_dummy",
    n_hops: int = 2,
) -> pd.DataFrame:
    """Compute attachment-point SMARTS for every unique (component, SMILES) pair.

    Linkers produce two rows — one per attachment point (cap1 → warhead side,
    cap2 → E3 side). ``component_id`` and ``protac_ids`` (if present in ``df``)
    are carried through into every output row. Unknown component types are
    recorded with an error.

    Args:
        df:            DataFrame with at least ``component_col``, ``smiles_col``,
                       and optionally ``component_id`` and ``protac_ids``.
        component_col: Column identifying the component type ("warhead", "linker", "e3").
        smiles_col:    Column with the component SMILES (including dummy atoms).
        n_hops:        Bond-distance radius passed to :func:`attachment_environment`.

    Returns:
        DataFrame with columns: components, component_smiles_with_dummy,
        component_id, protac_ids, attachment_id, attachment_smarts, error.
    """
    has_ids = "component_id" in df.columns and "protac_ids" in df.columns
    rows = []

    for _, row in df.dropna(subset=[smiles_col]).drop_duplicates(subset=[component_col, smiles_col]).iterrows():
        component = str(row[component_col]).strip()
        smiles    = row[smiles_col]
        id_fields = {
            "component_id":      row.get("component_id"),
            "component_hash_id": row.get("component_hash_id"),
            "protac_ids":        row.get("protac_ids"),
        } if has_ids else {}

        if component not in COMPONENT_MAP:
            rows.append({
                "components": component,
                "component_smiles_with_dummy": smiles,
                **id_fields,
                "attachment_id": pd.NA,
                "attachment_smarts": pd.NA,
                "error": f"unknown component type: {component}",
            })
            continue

        for label, att_id in COMPONENT_MAP[component]:
            base = {"components": label, "component_smiles_with_dummy": smiles, **id_fields, "attachment_id": att_id}
            mol, err = safe_mol(smiles)
            if mol is None:
                rows.append({**base, "attachment_smarts": pd.NA, "error": err})
                continue
            try:
                smarts = attachment_environment(mol, attachment_id=att_id, n_hops=n_hops)
                if smarts is None:
                    rows.append({**base, "attachment_smarts": pd.NA,
                                 "error": f"attachment [*:{att_id}] not found or failed"})
                else:
                    rows.append({**base, "attachment_smarts": smarts, "error": ""})
            except Exception as exc:
                rows.append({**base, "attachment_smarts": pd.NA, "error": str(exc)})

    return pd.DataFrame(rows)


def classify_environments(df: pd.DataFrame) -> pd.DataFrame:
    """Add an ``environment`` column by looking up each SMARTS in SMARTS_TO_ENVIRONMENT.

    Args:
        df: DataFrame with an ``attachment_smarts`` column.

    Returns:
        Copy of `df` with an added ``environment`` column.
    """
    df = df.copy()
    df["environment"] = df["attachment_smarts"].map(
        lambda s: SMARTS_TO_ENVIRONMENT.get(s, FALLBACK_ENVIRONMENT)
    )
    return df


# ── Visualization ─────────────────────────────────────────────────────────────

COMPONENT_ORDER  = ["warhead", "linker_cap1", "linker_cap2", "e3"]
COMPONENT_LABELS = {"warhead": "Warhead", "linker_cap1": "Linker\ncap1",
                    "linker_cap2": "Linker\ncap2", "e3": "E3 ligand"}
ENV_ORDER  = ["Aliphatic N", "Aromatic C", "Carbonyl C", "Aliphatic C", "Oxygen", "others"]
ENV_COLORS = {
    "Aliphatic N": "#A89CC8", "Aromatic C":  "#B8AED8", "Carbonyl C":  "#D4724A",
    "Aliphatic C": "#A0A090", "Oxygen":      "#4AAA80", "others":      "#C8C8B8",
}


def plot_env_summary(df: pd.DataFrame, output_path: Path) -> None:
    """Save a lollipop-bar figure of attachment-environment percentages per component.

    Args:
        df:          Classified attachment DataFrame (must have ``components`` and
                     ``environment`` columns).
        output_path: Destination PNG file.
    """
    df = df[df["components"].isin(COMPONENT_ORDER)].copy()

    summary = df.groupby(["components", "environment"]).size().reset_index(name="count")
    summary["percent"] = summary["count"] / summary.groupby("components")["count"].transform("sum") * 100

    pivot = (
        summary.pivot(index="environment", columns="components", values="percent")
        .fillna(0)
        .reindex(index=ENV_ORDER, columns=COMPONENT_ORDER)
        .fillna(0)
    )

    ROW_H, HDR_H, FIG_W = 1.25, 0.8, 16.0
    FIG_H   = HDR_H + len(ENV_ORDER) * ROW_H + 0.2
    max_pct = pivot.values.max() or 1

    COL_ENV  = 0.25
    COL_COMP = [3.2, 5.7, 8.2, 10.7]
    BAR_MAX_W = 1.8

    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="white")
    ax  = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, FIG_W)
    ax.set_ylim(0, FIG_H)
    ax.axis("off")

    def row_y(i: int) -> float:
        return FIG_H - HDR_H - i * ROW_H

    # Header
    hcy = FIG_H - HDR_H / 2
    for label, x in zip(
        ["Environment"] + [COMPONENT_LABELS[c] for c in COMPONENT_ORDER],
        [COL_ENV] + COL_COMP,
    ):
        ax.text(x, hcy, label, fontsize=10, fontweight="bold", va="center", color="#444444")

    # Rows
    for g, env in enumerate(ENV_ORDER):
        row_bot = row_y(g + 1)
        cy    = row_bot + ROW_H / 2
        color = ENV_COLORS.get(env, "#AAAAAA")

        if g % 2 == 0:
            ax.add_patch(plt.Rectangle((0, row_bot), FIG_W, ROW_H, color="#F7F7F5", zorder=0))

        ax.text(COL_ENV, cy, env, fontsize=16, fontweight="bold", va="center", color="#222222")

        for i, comp in enumerate(COMPONENT_ORDER):
            pct = pivot.loc[env, comp] if env in pivot.index else 0
            bw  = (pct / max_pct) * BAR_MAX_W
            bh  = ROW_H * 0.28
            ax.add_patch(plt.Rectangle((COL_COMP[i], cy - bh / 2), bw, bh, color=color, zorder=2))
            ax.text(COL_COMP[i] + bw + 0.08, cy,
                    f"{pct:.0f}%" if pct >= 1 else "<1%",
                    fontsize=14, va="center", color="#666666")

        ax.plot([0.1, FIG_W - 0.1], [row_bot, row_bot], color="#DDDDDD", linewidth=0.6)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def build_component_table(df: pd.DataFrame) -> pd.DataFrame:
    """Melt the three component columns into a long-format table with IDs and PROTAC membership.

    Groups by (smiles, component_id, component_hash_id) so each unique component
    gets one row, with ``protac_ids`` storing every PROTAC that contains it as a
    comma-separated string (sorted, no spaces).

    Output columns: components, component_smiles_with_dummy, component_id,
    component_hash_id, protac_ids.

    Args:
        df: Master PROTAC table with `warhead_smiles`/`linker_smiles`/
            `e3_ligase_ligand_smiles`, their `*_id`/`*_hash_id` columns, and
            `protac_id`.

    Returns:
        Long-format DataFrame, one row per unique component.
    """
    col_map = {
        "warhead_smiles":          ("warhead", "warhead_id",          "warhead_hash_id"),
        "linker_smiles":           ("linker",  "linker_id",           "linker_hash_id"),
        "e3_ligase_ligand_smiles": ("e3",      "e3_ligase_ligand_id", "e3_ligase_ligand_hash_id"),
    }
    parts = []
    for smiles_col, (label, id_col, hash_id_col) in col_map.items():
        part = (
            df[[smiles_col, id_col, hash_id_col, "protac_id"]]
            .dropna(subset=[smiles_col])
            .groupby([smiles_col, id_col, hash_id_col], dropna=False)["protac_id"]
            .apply(lambda ids: ",".join(str(i) for i in sorted(set(ids))))
            .reset_index()
            .rename(columns={
                smiles_col:   "component_smiles_with_dummy",
                id_col:       "component_id",
                hash_id_col:  "component_hash_id",
                "protac_id":  "protac_ids",
            })
            .assign(components=label)
        )
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the attachment-environment analysis CLI.

    Returns:
        Parsed arguments namespace.
    """
    p = argparse.ArgumentParser(description="Attachment-point environment analysis.")
    p.add_argument("--input",  type=Path, default=_PROJECT_ROOT / "data/processed/protac_smiles_master_std.csv",
                   help="Master PROTAC table (build_protac_master_table output).")
    p.add_argument("--output", type=Path, default=_PROJECT_ROOT / "data/processed/analysis/attachment_env.csv",
                   help="Classified attachment environments CSV.")
    p.add_argument("--plot",   type=Path, default=_PROJECT_ROOT / "figures/attachment_env.png",
                   help="Output figure path.")
    p.add_argument("--n-hops", type=int,  default=2,
                   help="Bond-distance radius around the attachment dummy atom.")
    return p.parse_args()


def main(input_path: Path, output_path: Path, plot_path: Path, n_hops: int) -> None:
    """Run the full attachment-environment analysis and write CSV + figure outputs.

    Args:
        input_path: Master PROTAC table CSV (build_protac_master_table output).
        output_path: Destination for the classified attachment environments CSV.
        plot_path: Destination for the summary figure PNG.
        n_hops: Bond-distance radius around each attachment dummy atom.
    """
    print("Loading PROTAC master file...")
    df_raw = pd.read_csv(input_path)

    df = build_component_table(df_raw)
    print(f"Unique components: {len(df)}")

    attachment_df  = build_attachment_analysis(df, n_hops=n_hops)
    classified_df  = classify_environments(attachment_df)

    others = classified_df[classified_df["environment"] == FALLBACK_ENVIRONMENT]["attachment_smarts"].value_counts()
    if not others.empty:
        print("\nUnclassified SMARTS (consider adding to SMARTS_TO_ENVIRONMENT):")
        print(others.to_string())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    classified_df.to_csv(output_path, index=False)
    print(f"\nSaved → {output_path}  ({len(classified_df)} rows)")

    plot_env_summary(classified_df, plot_path)
    print(f"Saved plot → {plot_path}")

    print("\nPreview:")
    print(classified_df[["components", "component_id", "component_hash_id", "protac_ids", "attachment_id", "attachment_smarts", "environment"]].head(10).to_string(index=False))


if __name__ == "__main__":
    args = parse_args()
    main(args.input, args.output, args.plot, args.n_hops)
