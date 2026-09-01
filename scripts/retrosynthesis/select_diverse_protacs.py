"""
select_diverse_protacs.py
─────────────────────────
Selects the top-k most diverse PROTACs from the master CSV, excluding any
molecules already present in a pre-existing selection list (e.g. top100).
The algorithm is seeded with the excluded molecules so new picks are
maximally diverse from both the existing list AND each other.

Diversity criteria:
1. Molecular fingerprint diversity  (Morgan FP + MaxMin picking)
2. Functional group diversity       (custom FG profile dissimilarity)
3. Component-level diversity        (similarity-based warhead / E3-ligase / linker discrimination)

Input CSV file
--------------
The file must contain a column with the standardized PROTAC SMILES, called protac_smiles_std.
Otherwise, it looks for the column protac_smiles

Output
------
- CSV: saved in the path data/processed, named top_<k>_diverse_protacs.csv
       Columns: protac_id, datasource, protac_smiles, protac_smiles_index, warhead_smiles,
       linker_smiles, e3_ligase_ligand_smiles, protac_canon_smiles, protac_smiles_std,
       protac_std_id, selection_rank
       The output molecules are ranked based on the diversity criteria.

Usage
-----
    python select_diverse_protacs.py --k 100 [--exclude existing.csv] [--output selected.csv]

Arguments:
    --master          Path to master PROTAC CSV (default: protac_smiles_master_std.csv)
    --exclude         Path to CSV of already-selected molecules to exclude and seed from
                      (default: top100_diverse_protacs.csv). Pass an empty string to skip.
    --sep             Delimiter for CSV files (default: auto-detect)
    --k               Number of new PROTACs to select (default: 20)
    --output          Output CSV path (default: top_k_diverse_protacs.csv)
    --seed            Random seed for reproducibility (default: 42)
    --fp_weight       Weight for fingerprint diversity component (default: 0.5)
    --fg_weight       Weight for functional-group diversity component (default: 0.3)
    --comp_weight     Weight for component uniqueness penalty (default: 0.2)
    --comp_threshold  Tanimoto threshold below which two components are considered
                      distinct (default: 0.85). Higher = stricter deduplication.
"""

import argparse
import sys
import warnings
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd
from rdkit import DataStructs

from retrotac.chem_utils import (
    canon_smiles,
    fg_vector,
    jaccard_fg,
    morgan_fp,
    smiles_to_component_fp,
    smiles_to_mol,
    tanimoto_distance,
)

warnings.filterwarnings("ignore")

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def read_csv_auto(path: Any, sep: Optional[str] = None) -> pd.DataFrame:
    """Read a CSV, auto-detecting the delimiter from the first line when *sep* is None.

    Args:
        path: File path passed to ``pd.read_csv``.
        sep: Explicit delimiter; when None, the first line is inspected to choose
            between ``,`` and ``;``.

    Returns:
        Loaded DataFrame.
    """
    if sep:
        return pd.read_csv(path, sep=sep)
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        first_line = f.readline()
    detected = ";" if first_line.count(";") >= first_line.count(",") else ","
    return pd.read_csv(path, sep=detected)


def component_fps_for_row(
    row: Any,
) -> Tuple[Optional[Any], Optional[Any], Optional[Any]]:
    """Compute Morgan fingerprints for all three PROTAC components in a DataFrame row.

    Args:
        row: Mapping with keys ``warhead_smiles``, ``e3_ligase_ligand_smiles``,
            and ``linker_smiles``.

    Returns:
        Tuple of ``(warhead_fp, e3_fp, linker_fp)``; unparseable components yield None.
    """
    return (
        smiles_to_component_fp(row.get("warhead_smiles", "")),
        smiles_to_component_fp(row.get("e3_ligase_ligand_smiles", "")),
        smiles_to_component_fp(row.get("linker_smiles", "")),
    )


def is_novel_component(
    fp: Optional[Any], used_fps: List[Any], threshold: float = 0.85
) -> bool:
    """Check whether *fp* differs from all previously selected component fingerprints.

    Args:
        fp: Morgan fingerprint of the candidate component; treated as novel when None.
        used_fps: Fingerprints of already-selected components of the same type.
        threshold: Tanimoto similarity ceiling; a component is novel only when its
            similarity to every entry in *used_fps* is strictly below this value.

    Returns:
        True if novel (or fp is None / used_fps is empty), False otherwise.
    """
    if fp is None or not used_fps:
        return True
    return all(
        DataStructs.TanimotoSimilarity(fp, u) < threshold
        for u in used_fps
        if u is not None
    )


def select_diverse(
    df: pd.DataFrame,
    k: int,
    seed_fps: List[Any],
    seed_fgvs: List[np.ndarray],
    seed_comp_fps: List[Tuple[Optional[Any], Optional[Any], Optional[Any]]],
    fp_w: float = 0.5,
    fg_w: float = 0.3,
    comp_w: float = 0.2,
    comp_threshold: float = 0.85,
    seed: int = 42,
) -> List[int]:
    """Greedy MaxMin diverse subset selection, pre-seeded with already-chosen molecules.

    Scores each candidate by a weighted sum of (1) Tanimoto distance to the nearest
    already-selected full-PROTAC fingerprint, (2) Jaccard FG distance, and (3) a
    component-novelty bonus that rewards warhead/E3/linker chemotypes not yet seen.
    The highest-scoring candidate is added each round, distances are updated, and the
    component FP banks are extended.

    Args:
        df: Candidate DataFrame with pre-computed ``_fp``, ``_fg``, and ``_comp_fps``
            columns (produced by the parsing block in ``main``).
        k: Number of molecules to select.
        seed_fps: Full-PROTAC Morgan FPs of already-selected molecules (diversity seed).
        seed_fgvs: FG vectors of already-selected molecules.
        seed_comp_fps: ``(warhead_fp, e3_fp, linker_fp)`` tuples for already-selected molecules.
        fp_w: Weight for the fingerprint diversity component.
        fg_w: Weight for the functional-group diversity component.
        comp_w: Weight for the component-novelty bonus.
        comp_threshold: Tanimoto similarity ceiling for component novelty.
        seed: RNG seed used only when no seed molecules are provided.

    Returns:
        Row indices into *df* of the *k* selected molecules, in selection order.
    """
    n = len(df)
    if n < k:
        sys.exit(f"ERROR: Only {n} candidate molecules available but k={k} requested.")

    fps = df["_fp"].tolist()
    fgvs = np.stack(df["_fg"].values)
    comp_fps = df["_comp_fps"].tolist()

    if seed_fps:
        print(f"  Seeding distances from {len(seed_fps)} existing molecules …")
        dist_fp = np.array([min(tanimoto_distance(fp, s) for s in seed_fps) for fp in fps])
        dist_fg = np.array([min(jaccard_fg(fgvs[i], s) for s in seed_fgvs) for i in range(n)])
        selected = []

        used_warhead_fps = [c[0] for c in seed_comp_fps if c[0] is not None]
        used_e3_fps = [c[1] for c in seed_comp_fps if c[1] is not None]
        used_linker_fps = [c[2] for c in seed_comp_fps if c[2] is not None]

    else:
        rng = np.random.default_rng(seed)
        start = int(rng.integers(0, n))
        selected = [start]

        dist_fp = np.array([tanimoto_distance(fps[i], fps[start]) for i in range(n)])
        dist_fg = np.array([jaccard_fg(fgvs[i], fgvs[start]) for i in range(n)])

        wh0, e3_0, lnk0 = comp_fps[start]
        used_warhead_fps = [wh0] if wh0 is not None else []
        used_e3_fps = [e3_0] if e3_0 is not None else []
        used_linker_fps = [lnk0] if lnk0 is not None else []

    while len(selected) < k:
        # Each novel component contributes 1/3 to the bonus (range 0.0–1.0).
        comp_bonus = np.array(
            [
                (
                    is_novel_component(wh, used_warhead_fps, comp_threshold) +
                    is_novel_component(e3, used_e3_fps, comp_threshold) +
                    is_novel_component(lnk, used_linker_fps, comp_threshold)
                ) / 3.0
                for wh, e3, lnk in comp_fps
            ]
        )

        score = fp_w * dist_fp + fg_w * dist_fg + comp_w * comp_bonus
        if selected:
            score[selected] = -1.0

        best = int(np.argmax(score))
        selected.append(best)

        wh_best, e3_best, lnk_best = comp_fps[best]
        if wh_best is not None:
            used_warhead_fps.append(wh_best)
        if e3_best is not None:
            used_e3_fps.append(e3_best)
        if lnk_best is not None:
            used_linker_fps.append(lnk_best)

        for i in range(n):
            dist_fp[i] = min(dist_fp[i], tanimoto_distance(fps[i], fps[best]))
            dist_fg[i] = min(dist_fg[i], jaccard_fg(fgvs[i], fgvs[best]))

        if len(selected) % 5 == 0:
            print(f"  Selected {len(selected)}/{k} molecules …")

    return selected


def main() -> None:
    """Entry point: parse args, load data, run greedy selection, write output."""
    parser = argparse.ArgumentParser(description="Select top-k diverse PROTACs")
    parser.add_argument("--master", type=Path,
                        default=_PROJECT_ROOT / "data/processed/protac_smiles_master_std.csv",
                        help="Path to master PROTAC CSV.")
    parser.add_argument("--exclude", type=str, default=None,
                        help="CSV of already-selected molecules to exclude from pool "
                             "and use as diversity seed. Pass '' to skip.")
    parser.add_argument("--sep", type=str, default=None,
                        help="CSV delimiter (default: auto-detect)")
    parser.add_argument("--k", type=int, default=20,
                        help="Number of new PROTACs to select (default: 20)")
    parser.add_argument("--output", type=Path,
                        default=_PROJECT_ROOT / "data/processed/top_k_diverse_protacs.csv",
                        help="Output CSV path.")
    parser.add_argument("--smiles-col", type=str, default=None,
                        help="Name of the SMILES column in the master CSV (default: auto-detect).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp_weight", type=float, default=0.5)
    parser.add_argument("--fg_weight", type=float, default=0.3)
    parser.add_argument("--comp_weight", type=float, default=0.2)
    parser.add_argument("--comp_threshold", type=float, default=0.85,
                        help="Tanimoto similarity threshold for component deduplication. "
                             "Two components are treated as the same when similarity >= "
                             "this value (default: 0.85). Range: 0.0–1.0.")
    args = parser.parse_args()

    total = args.fp_weight + args.fg_weight + args.comp_weight
    if abs(total - 1.0) > 1e-3:
        print(f"⚠  Weights sum to {total:.3f} — normalising to 1.0")
        args.fp_weight /= total
        args.fg_weight /= total
        args.comp_weight /= total

    print(f"Loading master CSV: {args.master} …")
    master = read_csv_auto(args.master, sep=args.sep)
    print(f"  Loaded {len(master)} rows")

    if args.smiles_col:
        smi_col = args.smiles_col
    else:
        smi_col = "protac_smiles_std" if "protac_smiles_std" in master.columns else "protac_smiles"
    print(f"  Using SMILES column: '{smi_col}'")

    seed_fps, seed_fgvs, seed_comp_fps = [], [], []
    exclude_canon: set = set()

    if args.exclude is not None:
        print(f"Loading exclude list: {args.exclude} …")
        excl_df = read_csv_auto(args.exclude, sep=args.sep)
        smi_col_excl = "protac_smiles_std" if "protac_smiles_std" in excl_df.columns else "protac_smiles"

        n_excl = 0
        for _, row in excl_df.iterrows():
            smi = row[smi_col_excl]
            c = canon_smiles(smi)
            if c:
                exclude_canon.add(c)
            mol = smiles_to_mol(smi)
            if mol:
                seed_fps.append(morgan_fp(mol))
                seed_fgvs.append(fg_vector(mol))
                seed_comp_fps.append(component_fps_for_row(row))
                n_excl += 1

        print(f"  {n_excl} valid molecules in exclude list → will be removed from pool and used as seed")
    else:
        print("No exclude list provided — selecting from full master with random seed start")

    master["_canon"] = master[smi_col].apply(canon_smiles)
    df = master[~master["_canon"].isin(exclude_canon)].copy()
    df = df.drop(columns=["_canon"]).reset_index(drop=True)

    n_excluded = len(master) - len(df)
    print(f"  Excluded {n_excluded} molecules already in exclude list")
    print(f"  Candidate pool: {len(df)} PROTACs")

    print("Parsing SMILES …")
    df["_mol"] = df[smi_col].apply(smiles_to_mol)
    n_invalid = df["_mol"].isna().sum()
    if n_invalid:
        print(f"  Dropping {n_invalid} invalid SMILES")
    df = df[df["_mol"].notna()].reset_index(drop=True)
    print(f"  Valid candidates: {len(df)}")

    print("Computing Morgan fingerprints …")
    df["_fp"] = df["_mol"].apply(morgan_fp)
    print("Computing functional group vectors …")
    df["_fg"] = df["_mol"].apply(fg_vector)

    print("Computing component fingerprints (warhead / E3 ligand / linker) …")
    for col in ["warhead_smiles", "e3_ligase_ligand_smiles", "linker_smiles"]:
        if col not in df.columns:
            df[col] = ""
    df["_comp_fps"] = df.apply(component_fps_for_row, axis=1)

    n_missing_wh = df["_comp_fps"].apply(lambda t: t[0] is None).sum()
    n_missing_e3 = df["_comp_fps"].apply(lambda t: t[1] is None).sum()
    n_missing_lnk = df["_comp_fps"].apply(lambda t: t[2] is None).sum()
    if any([n_missing_wh, n_missing_e3, n_missing_lnk]):
        print("  ⚠  Components with unparseable SMILES (treated as novel):")
        print(f"     warhead={n_missing_wh}  |  E3={n_missing_e3}  |  linker={n_missing_lnk}")

    print(f"\nRunning greedy MaxMin selection (k={args.k}) …")
    print(f"  Weights       →  FP: {args.fp_weight:.2f}  |  FG: {args.fg_weight:.2f}  |  Component: {args.comp_weight:.2f}")
    print(f"  Comp threshold→  Tanimoto < {args.comp_threshold:.2f} → distinct component")

    indices = select_diverse(
        df, args.k,
        seed_fps=seed_fps, seed_fgvs=seed_fgvs, seed_comp_fps=seed_comp_fps,
        fp_w=args.fp_weight, fg_w=args.fg_weight, comp_w=args.comp_weight,
        comp_threshold=args.comp_threshold,
        seed=args.seed,
    )

    result = df.iloc[indices].copy()
    result["selection_rank"] = range(1, args.k + 1)
    drop_cols = [c for c in result.columns if c.startswith("_")]
    result.drop(columns=drop_cols, inplace=True)

    print("\n── Diversity Report ──────────────────────────────────────────────────")
    if "warhead_smiles" in result.columns:
        def count_unique_by_similarity(smiles_series: Any, threshold: float) -> int:
            """Count distinct component chemotypes by similarity clustering."""
            fps = [smiles_to_component_fp(s) for s in smiles_series]
            fps = [f for f in fps if f is not None]
            clusters: List[Any] = []
            for fp in fps:
                if not any(DataStructs.TanimotoSimilarity(fp, c) >= threshold for c in clusters):
                    clusters.append(fp)
            return len(clusters)

        n_wh = count_unique_by_similarity(result["warhead_smiles"], args.comp_threshold)
        n_e3 = count_unique_by_similarity(result["e3_ligase_ligand_smiles"], args.comp_threshold)
        n_lnk = count_unique_by_similarity(result["linker_smiles"], args.comp_threshold)
        print(f"  Unique warhead chemotypes  (Tanimoto < {args.comp_threshold:.2f}): {n_wh}")
        print(f"  Unique E3 ligand chemotypes(Tanimoto < {args.comp_threshold:.2f}): {n_e3}")
        print(f"  Unique linker chemotypes   (Tanimoto < {args.comp_threshold:.2f}): {n_lnk}")

    sel_mols = [smiles_to_mol(s) for s in result[smi_col]]
    sel_fps = [morgan_fp(mol) for mol in sel_mols if mol is not None]
    dists = [
        tanimoto_distance(sel_fps[i], sel_fps[j])
        for i in range(len(sel_fps))
        for j in range(i + 1, len(sel_fps))
    ]
    if dists:
        print(f"  Pairwise Tanimoto (new picks only)  →  "
              f"mean: {np.mean(dists):.3f}  |  min: {np.min(dists):.3f}  |  max: {np.max(dists):.3f}")

    if seed_fps and sel_fps:
        cross = [tanimoto_distance(fp, s) for fp in sel_fps for s in seed_fps]
        print(f"  Tanimoto vs excluded set            →  "
              f"mean: {np.mean(cross):.3f}  |  min: {np.min(cross):.3f}")
    print("─────────────────────────────────────────────────────────────────────\n")

    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    print(f"Saved {args.k} diverse PROTACs → {output_path}")


if __name__ == "__main__":
    main()
