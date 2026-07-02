"""
Stage 1, step 4 — cap open attachment points so components are valid molecules.

PROTAC-Splitter leaves a dummy atom ([*:n]) at each attachment point. Each dummy
is replaced by one of six small caps (H, OH, CH3, NH2, =O, COOH) chosen to
approximate local attachment chemistry. Warheads and E3 ligands have one attachment
point (6 caps each); linkers have two, so all 36 cap-pair combinations are enumerated.

I/O
---
    in :  data/processed/protac_smiles_master_std.csv
    out:  data/processed/component_capped.csv
          columns: component_id, component_smiles_with_dummy,
                   cap_type, cap_smiles, error
"""

import argparse
import sys
from itertools import product
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from rdkit import Chem, RDLogger

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

CAP_METHODS = ["H", "OH", "CH3", "NH2", "=O", "COOH"]

OUT_COLS = [
    "component_id", "component_smiles_with_dummy",
    "cap_type", "cap_smiles", "error",
]

# Disable RDKit warnings
RDLogger.DisableLog('rdApp.*')

# ── Chemistry helpers ─────────────────────────────────────────────────────────

def safe_mol(smiles: str) -> Chem.Mol:
    """Parse SMILES and return an RDKit Mol, raising ValueError on failure."""
    if pd.isna(smiles) or not str(smiles).strip():
        raise ValueError("empty SMILES")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"invalid SMILES: {smiles}")
    return mol


def add_cap(rw_mol: Chem.RWMol, neighbor_idx: int, method: str) -> None:
    """Attach a capping group to ``neighbor_idx`` in ``rw_mol`` (in-place).

    The dummy atom at the attachment point must already be removed before
    calling this function. H leaves no new atoms (implicit hydrogen).
    """
    if method == "H":
        return
    elif method == "CH3":
        c = rw_mol.AddAtom(Chem.Atom(6))
        rw_mol.AddBond(neighbor_idx, c, Chem.BondType.SINGLE)
    elif method == "OH":
        o = rw_mol.AddAtom(Chem.Atom(8))
        rw_mol.AddBond(neighbor_idx, o, Chem.BondType.SINGLE)
    elif method == "NH2":
        n = rw_mol.AddAtom(Chem.Atom(7))
        rw_mol.AddBond(neighbor_idx, n, Chem.BondType.SINGLE)
    elif method == "=O":
        o = rw_mol.AddAtom(Chem.Atom(8))
        rw_mol.AddBond(neighbor_idx, o, Chem.BondType.DOUBLE)
    elif method == "COOH":
        c  = rw_mol.AddAtom(Chem.Atom(6))
        o1 = rw_mol.AddAtom(Chem.Atom(8))
        o2 = rw_mol.AddAtom(Chem.Atom(8))
        rw_mol.AddBond(neighbor_idx, c,  Chem.BondType.SINGLE)
        rw_mol.AddBond(c, o1, Chem.BondType.DOUBLE)
        rw_mol.AddBond(c, o2, Chem.BondType.SINGLE)
    else:
        raise ValueError(f"unknown cap method: {method!r}")


def cap_component(smiles: str, caps: dict[int, str]) -> tuple[str, str, str]:
    """Replace dummy atoms with capping groups and return the resulting SMILES.

    Args:
        smiles: Component SMILES containing dummy atoms ([*:n]).
        caps:   Mapping of attachment-point map number → cap method, e.g.
                ``{1: "CH3"}`` for a warhead or ``{1: "H", 2: "OH"}`` for a linker.

    Returns:
        Tuple of (capped_smiles, canonical_smiles, error). On failure, the first
        two elements are ``pd.NA`` and error contains the exception message.

    Notes:
        Dummies are removed in descending index order so removing one atom does
        not shift the index of a not-yet-processed dummy.
    """
    try:
        mol = safe_mol(smiles)
        dummy_idx_by_map = {
            atom.GetAtomMapNum(): atom.GetIdx()
            for atom in mol.GetAtoms()
            if atom.GetAtomicNum() == 0
        }
        for map_num in caps:
            if map_num not in dummy_idx_by_map:
                raise ValueError(f"attachment [*:{map_num}] not found")

        rw_mol = Chem.RWMol(mol)
        for map_num, method in sorted(caps.items(), key=lambda kv: dummy_idx_by_map[kv[0]], reverse=True):
            dummy_idx   = dummy_idx_by_map[map_num]
            nbrs        = [n.GetIdx() for n in rw_mol.GetAtomWithIdx(dummy_idx).GetNeighbors()]
            if len(nbrs) != 1:
                raise ValueError(f"dummy atom [*:{map_num}] must have exactly one neighbor")
            neighbor_idx = nbrs[0]
            rw_mol.RemoveAtom(dummy_idx)
            if neighbor_idx > dummy_idx:
                neighbor_idx -= 1
            add_cap(rw_mol, neighbor_idx, method)

        capped = rw_mol.GetMol()
        Chem.SanitizeMol(capped)
        return (
            Chem.MolToSmiles(capped, canonical=False),
            Chem.MolToSmiles(capped, canonical=True),
            "",
        )
    except Exception as exc:
        return pd.NA, pd.NA, str(exc)


# ── Builder ───────────────────────────────────────────────────────────────────

def _cap_combos(attachment_maps: list[int]) -> list[dict[int, str]]:
    """All cap combinations for the given attachment-point map numbers.

    Single attachment point → 6 dicts; two attachment points → 36 dicts.
    """
    return [
        dict(zip(attachment_maps, combo))
        for combo in product(CAP_METHODS, repeat=len(attachment_maps))
    ]


def build_capped_df(
    df: pd.DataFrame,
    smiles_col: str,
    id_col: str,
    attachment_maps: list[int],
    component_name: str,
) -> pd.DataFrame:
    """Enumerate all capped variants for one component type.

    For each unique SMILES in ``smiles_col``, every cap combination is tried and
    the result is validated with RDKit. ``cap_type`` encodes the combination as a
    ``|``-joined string (e.g. ``"CH3|OH"`` for a linker capped with CH3 at [*:1]
    and OH at [*:2]).

    Args:
        df:               Source DataFrame (master table).
        smiles_col:       Column with component SMILES including dummy atoms.
        id_col:           Column with the component's sequential ID.
        attachment_maps:  Ordered list of attachment-point map numbers to cap
                          (``[1]`` for warhead, ``[2]`` for E3, ``[1, 2]`` for linker).
        component_name:   Label written to the ``component`` column.

    Returns:
        DataFrame with columns matching ``OUT_COLS``.
    """
    unique = df[[smiles_col, id_col]].dropna(subset=[smiles_col]).drop_duplicates(subset=[smiles_col])
    rows = []
    for _, row in tqdm(unique.iterrows(), total=len(unique), desc=f"Capping {component_name} components"):
        smiles  = row[smiles_col]
        comp_id = row[id_col]
        for caps in _cap_combos(attachment_maps):
            capped, canon, error = cap_component(smiles, caps)
            rows.append({
                "component_id":                comp_id,
                "component_smiles_with_dummy": smiles,
                "cap_type":                    "|".join(caps[m] for m in sorted(caps)),
                "cap_smiles":                  canon,
                "error":                       error,
            })
    return pd.DataFrame(rows, columns=OUT_COLS)


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cap component attachment points.")
    p.add_argument("--input",  type=Path, default=_PROJECT_ROOT / "data/processed/protac_smiles_master_std.csv",
                   help="Master PROTAC table (build_protac_master_table output).")
    p.add_argument("--output", type=Path, default=_PROJECT_ROOT / "data/processed/component_capped.csv",
                   help="Destination CSV.")
    p.add_argument("--log-errors", action="store_true",
                   help="Keep failed rows and include the 'error' column in the output. "
                        "By default, rows with no valid cap_smiles are dropped and the column is omitted.")
    return p.parse_args()


def main(input_path: Path, output_path: Path, log_errors: bool = False) -> None:
    df = pd.read_csv(input_path)

    parts = [
        build_capped_df(df, "warhead_smiles", "warhead_hash_id", [1], "warhead"),
        build_capped_df(df, "e3_ligase_ligand_smiles", "e3_ligase_ligand_hash_id", [2], "e3_ligase_ligand"),
        build_capped_df(df, "linker_smiles", "linker_hash_id", [1, 2], "linker"),
    ]
    all_df = pd.concat(parts, ignore_index=True)

    n_failed = all_df["cap_smiles"].isna().sum()
    if log_errors:
        print(f"Rows with capping errors: {n_failed}")
    else:
        all_df = all_df.dropna(subset=["cap_smiles"]).drop(columns=["error"])
        print(f"Dropped {n_failed} rows with capping errors.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    all_df.to_csv(output_path, index=False)
    print(f"Saved → {output_path}  ({len(all_df)} rows)")


if __name__ == "__main__":
    args = parse_args()
    main(args.input, args.output, log_errors=args.log_errors)
