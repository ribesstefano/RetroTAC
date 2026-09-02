"""
scaffold_analysis.py
─────────────────────────
Diagnose which SMILES column (or combination of columns) should define the grouping
key for scaffold-grouped cross-validation, by measuring -- on a real dataset -- how
much each candidate key constrains the split and how much label signal it carries.

Motivating problem: for multi-component molecules (e.g. PROTACs = warhead + linker +
E3 ligand) a Murcko/generic scaffold of the *whole* molecule is so specific that
near-identical analogues (one CH2 more in the linker) get different scaffolds and land
in different folds. Grouping on the whole molecule then looks rigorous while leaving
the shared sub-structure on both sides of every split.

Composite keys
--------------
A key may combine several columns, and the two ways of combining pull in OPPOSITE
directions -- this is the main thing this script is for:

  COL_A+COL_B  (product) group = the (scaffold_A, scaffold_B) pair. This partition is
                         FINER than either column alone, so it leaks MORE per component:
                         one warhead paired with 5 different E3 ligands becomes 5 groups
                         that scatter across folds, putting that warhead on both sides.
  COL_A&COL_B  (union)   group = a connected component of the graph linking rows that
                         share scaffold_A OR scaffold_B. This partition is COARSER than
                         either column alone and is the only one that actually guarantees
                         neither constituent scaffold crosses a fold boundary. Watch for
                         percolation: one giant component makes folds unbalanceable.

Six reports, all printed; each is independently useful:

  1. Overview          -- rows, non-null and unique counts per underlying column.
  2. Acyclic fallback  -- `get_scaffold` returns the *input SMILES* when a molecule has
                          no ring system, so grouping silently degenerates to exact
                          string matching. A high rate here (typical for PEG/alkyl
                          linkers) disqualifies a column as a grouping key.
  3. Group counts      -- unique groups under generic vs Murcko scaffolds. Fewer groups
                          = coarser = a stronger constraint on the split.
  4. Group sizes       -- size distribution + whether n_folds can be balanced at all
                          (a group larger than one fold cannot be, by construction).
  5. Measured leakage  -- runs the REAL splitter (`get_fold_indices`) once per candidate
                          key and reports, averaged over folds, the fraction of held-out
                          rows whose *other* components also occur in training. This is
                          the number that matters: it quantifies the memorisation channel
                          each grouping choice leaves open.
  6. eta^2             -- fraction of each target's variance explained by group
                          membership. High eta^2 on a key means the label is largely
                          determined by that key, so leaving it unconstrained (report 5)
                          is what lets a model memorise instead of generalise.

Read reports 5 and 6 together: a key that is both high-eta^2 and high-leakage under your
current grouping is the one you should be grouping on.

Usage
-----
    # baseline: the four single keys
    python scripts/dataset/scaffold_analysis.py data/routes/routes_scored_with_components.csv

    # add every pairwise + three-way combination of the component keys, both modes
    python scripts/dataset/scaffold_analysis.py data/routes/routes_scored_with_components.csv \\
        --combos all --combo-keys warhead linker e3 --combo-mode both \\
        --targets synthesizability sa_score struct_n_steps \\
        --output-dir outputs/analysis/scaffold_choice

    # one explicit composite key
    python scripts/dataset/scaffold_analysis.py data/routes/routes_scored_with_components.csv \\
        --group-cols whole=smiles warhead=wh_smiles wh_and_e3=wh_smiles&e3_smiles

Arguments:
    input_csv         CSV to analyse; must contain every column named in --group-cols.
    --group-cols      NAME=COLUMN pairs defining the candidate grouping keys. COLUMN may
                      be a single column, COL_A+COL_B (product) or COL_A&COL_B (union) --
                      see "Composite keys" above. Quote '&' in most shells.
                      (default: whole=smiles warhead=wh_smiles linker=linker_smiles
                      e3=e3_smiles).
    --combos          none | pairs | all -- auto-generate combinations of the single-column
                      keys: "pairs" adds every 2-subset, "all" adds every subset of size
                      >= 2 (default: none).
    --combo-keys      Which single-column keys participate in --combos (default: all of
                      them; usually you want to exclude the whole-molecule key).
    --combo-mode      product | union | both -- which combination semantics --combos
                      generates (default: both).
    --targets         Target columns for the eta^2 report (default: every numeric
                      column with at least --min-target-rows non-null values).
    --scaffold-mode   generic | murcko | both -- which scaffold flavour to report and
                      to use for the leakage/eta^2 reports (default: both, which
                      reports both in report 3 and uses --primary-mode elsewhere).
    --primary-mode    generic | murcko -- the flavour used for reports 4-6 when
                      --scaffold-mode is "both" (default: generic, matching
                      `get_scaffold`'s current default).
    --n-folds         Folds to simulate in the leakage report (default: 5).
    --seed            Seed for the simulated split (default: 42).
    --min-target-rows Minimum non-null rows for a column to be auto-selected as a
                      target (default: 1000).
    --output-dir      If given, also write group_stats.csv, leakage.csv and
                      eta_squared.csv here.
"""

import argparse
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol

from retrotac.chem_utils import get_scaffold, smiles_to_mol

DEFAULT_GROUP_COLS = [
    "whole=smiles",
    "warhead=wh_smiles",
    "linker=linker_smiles",
    "e3=e3_smiles",
]

# a key is (mode, [columns]); mode is "single", "product" or "union"
KeySpec = Tuple[str, List[str]]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("input_csv", type=str, help="CSV to analyse.")
    p.add_argument(
        "--group-cols",
        nargs="+",
        default=DEFAULT_GROUP_COLS,
        metavar="NAME=COLUMN",
        help="Candidate grouping keys; COLUMN may be COL, COL_A+COL_B or COL_A&COL_B.",
    )
    p.add_argument(
        "--combos",
        choices=["none", "pairs", "all"],
        default="none",
        help="Auto-generate combinations of the single-column keys (default: none).",
    )
    p.add_argument(
        "--combo-keys",
        nargs="*",
        default=None,
        help="Single-column key names to combine (default: all of them).",
    )
    p.add_argument(
        "--combo-mode",
        choices=["product", "union", "both"],
        default="both",
        help="Combination semantics generated by --combos (default: both).",
    )
    p.add_argument(
        "--targets",
        nargs="*",
        default=None,
        help="Target columns for the eta^2 report (default: auto-detect numeric columns).",
    )
    p.add_argument(
        "--scaffold-mode",
        choices=["generic", "murcko", "both"],
        default="both",
        help="Which scaffold flavour to report (default: both).",
    )
    p.add_argument(
        "--primary-mode",
        choices=["generic", "murcko"],
        default="generic",
        help="Flavour used for reports 4-6 when --scaffold-mode is 'both' (default: generic).",
    )
    p.add_argument("--n-folds", type=int, default=5, help="Folds to simulate (default: 5).")
    p.add_argument("--seed", type=int, default=42, help="Seed for the simulated split (default: 42).")
    p.add_argument(
        "--min-target-rows",
        type=int,
        default=1000,
        help="Minimum non-null rows for auto-selecting a target column (default: 1000).",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="If given, also write group_stats.csv, leakage.csv and eta_squared.csv here.",
    )
    return p.parse_args()


def parse_group_cols(pairs: List[str], df: pd.DataFrame) -> Dict[str, KeySpec]:
    """Turn NAME=COLUMN strings into an ordered {name: (mode, columns)} mapping.

    COLUMN is one column, "COL_A+COL_B" (product) or "COL_A&COL_B" (union); the two
    separators must not be mixed within one key.

    Args:
        pairs: Strings of the form "name=column[{+,&}column...]".
        df: DataFrame the columns must exist in.

    Returns:
        Mapping from display name to (mode, list of column names).

    Raises:
        ValueError: If a pair is malformed, mixes separators, or names a missing column.
    """
    out: Dict[str, KeySpec] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"--group-cols entry '{pair}' is not of the form NAME=COLUMN.")
        name, spec = pair.split("=", 1)
        if "+" in spec and "&" in spec:
            raise ValueError(f"--group-cols entry '{pair}' mixes '+' and '&'; use one or the other.")
        if "+" in spec:
            mode, cols = "product", spec.split("+")
        elif "&" in spec:
            mode, cols = "union", spec.split("&")
        else:
            mode, cols = "single", [spec]
        for col in cols:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' (key '{name}') not found. Available: {df.columns.tolist()}")
        out[name] = (mode, cols)
    return out


def add_auto_combos(keys: Dict[str, KeySpec], combos: str, combo_keys: Optional[List[str]],
                    combo_mode: str) -> Dict[str, KeySpec]:
    """Extend `keys` with auto-generated combinations of its single-column keys.

    Args:
        keys: Parsed key mapping to extend.
        combos: "none", "pairs" (2-subsets) or "all" (every subset of size >= 2).
        combo_keys: Single-column key names to combine, or None for all of them.
        combo_mode: "product", "union" or "both".

    Returns:
        A new mapping with the generated keys appended (originals kept, in order).

    Raises:
        ValueError: If `combo_keys` names something that is not a single-column key.
    """
    if combos == "none":
        return keys
    singles = [n for n, (mode, _) in keys.items() if mode == "single"]
    if combo_keys is not None:
        unknown = [k for k in combo_keys if k not in singles]
        if unknown:
            raise ValueError(f"--combo-keys {unknown} are not single-column keys. Available: {singles}")
        singles = [k for k in singles if k in combo_keys]
    if len(singles) < 2:
        raise ValueError(f"--combos needs at least 2 single-column keys, got {singles}.")

    sizes = [2] if combos == "pairs" else range(2, len(singles) + 1)
    modes = ["product", "union"] if combo_mode == "both" else [combo_mode]
    sep = {"product": "+", "union": "&"}

    out = dict(keys)
    for size in sizes:
        for subset in combinations(singles, size):
            for mode in modes:
                name = sep[mode].join(subset)
                cols = [keys[s][1][0] for s in subset]
                out[name] = (mode, cols)
    return out


def select_targets(df: pd.DataFrame, explicit: Optional[List[str]], keys: Dict[str, KeySpec],
                   min_rows: int) -> List[str]:
    """Choose the target columns for the eta^2 report.

    Args:
        df: Source DataFrame.
        explicit: User-supplied target names, or None to auto-detect.
        keys: Grouping-key mapping, whose columns are never targets.
        min_rows: Minimum non-null rows for an auto-detected target.

    Returns:
        List of target column names (possibly empty).

    Raises:
        ValueError: If an explicitly requested column is missing.
    """
    if explicit is not None:
        missing = [t for t in explicit if t not in df.columns]
        if missing:
            raise ValueError(f"Target column(s) {missing} not found. Available: {df.columns.tolist()}")
        return explicit
    skip = underlying_columns(keys)
    return [
        c for c in df.columns
        if c not in skip
        and pd.api.types.is_numeric_dtype(df[c])
        and df[c].notna().sum() >= min_rows
        and df[c].nunique() > 1
    ]


def underlying_columns(keys: Dict[str, KeySpec]) -> List[str]:
    """List the distinct DataFrame columns referenced by any key, in first-seen order.

    Args:
        keys: Grouping-key mapping.

    Returns:
        Ordered list of unique column names.
    """
    seen: List[str] = []
    for _, cols in keys.values():
        for col in cols:
            if col not in seen:
                seen.append(col)
    return seen


def _union_components(parts: List[pd.Series]) -> pd.Series:
    """Label each row by the connected component of the "shares any scaffold" graph.

    Rows sharing a value in ANY of `parts` end up in the same component, so no
    constituent scaffold can straddle a group boundary (unlike a product key).

    Args:
        parts: Per-row scaffold Series, all sharing one index; NaN never links rows.

    Returns:
        Series of component-root labels, aligned to the inputs' index.
    """
    n = len(parts[0])
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:               # iterative + path halving: no recursion limit
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for series in parts:
        first: Dict[object, int] = {}
        for pos, value in enumerate(series.to_numpy()):
            if pd.isna(value):
                continue
            if value in first:
                union(first[value], pos)
            else:
                first[value] = pos
    return pd.Series([find(i) for i in range(n)], index=parts[0].index).astype(str)


def compute_scaffold_keys(df: pd.DataFrame, keys: Dict[str, KeySpec],
                          generic: bool) -> Dict[str, pd.Series]:
    """Build the per-row group label for every key, caching scaffolds per unique SMILES.

    Args:
        df: Source DataFrame.
        keys: Grouping-key mapping.
        generic: True for element-agnostic scaffolds, False for Murcko.

    Returns:
        Mapping from key name to a Series of per-row group labels (NaN preserved for
        single/product keys whose source SMILES was NaN).
    """
    # scaffold computation is the expensive step -- do it once per unique SMILES,
    # then reuse across every key that references the column
    per_col: Dict[str, pd.Series] = {}
    for col in underlying_columns(keys):
        mapping = {s: get_scaffold(s, generic=generic) for s in df[col].dropna().unique()}
        per_col[col] = df[col].map(mapping)

    out: Dict[str, pd.Series] = {}
    for name, (mode, cols) in keys.items():
        parts = [per_col[c] for c in cols]
        if mode == "single":
            out[name] = parts[0]
        elif mode == "product":
            joined = parts[0].astype(str)
            for part in parts[1:]:
                joined = joined + "||" + part.astype(str)
            # a product key is undefined if any constituent is missing
            out[name] = joined.where(pd.concat(parts, axis=1).notna().all(axis=1))
        else:
            out[name] = _union_components(parts)
    return out


def report_overview(df: pd.DataFrame, keys: Dict[str, KeySpec]) -> None:
    """Print row counts, per-column non-null/unique counts, and the key definitions.

    Args:
        df: Source DataFrame.
        keys: Grouping-key mapping.
    """
    print("=" * 86)
    print("1. OVERVIEW")
    print("=" * 86)
    print(f"rows: {len(df)}")
    for col in underlying_columns(keys):
        n_null = df[col].isna().sum()
        n_uniq = df[col].nunique()
        dupes = df[col].notna().sum() - n_uniq
        print(f"  {col:<18s} non-null={df[col].notna().sum():>7d}  unique={n_uniq:>7d}  "
              f"exact-duplicate rows={dupes:>6d} ({100 * dupes / max(len(df), 1):.1f}%)"
              + (f"  MISSING={n_null}" if n_null else ""))
    sep = {"single": "", "product": " + ", "union": " & "}
    print("\ncandidate keys:")
    for name, (mode, cols) in keys.items():
        note = {"single": "", "product": "  (product: FINER than its parts)",
                "union": "  (union: COARSER than its parts)"}[mode]
        print(f"  {name:<22s} = {sep[mode].join(cols)}{note}")
    complete = df.dropna(subset=underlying_columns(keys))
    print(f"\ncomplete-case rows (all columns present): {len(complete)} "
          f"({100 * len(complete) / max(len(df), 1):.1f}%)")


def report_acyclic_fallback(df: pd.DataFrame, keys: Dict[str, KeySpec]) -> None:
    """Print how often `get_scaffold` falls back to the raw SMILES, per column.

    A molecule with no ring system has an empty Murcko scaffold, so `get_scaffold`
    returns the input SMILES unchanged and "grouping" becomes exact string matching.

    Args:
        df: Source DataFrame.
        keys: Grouping-key mapping.
    """
    print("\n" + "=" * 86)
    print("2. ACYCLIC FALLBACK  (no ring system -> scaffold == raw SMILES)")
    print("=" * 86)
    print("A high rate disqualifies the column: grouping degenerates to exact matching,")
    print("so homologues (PEG-3 vs PEG-4) become different groups and split across folds.\n")
    for col in underlying_columns(keys):
        uniq = df[col].dropna().unique()
        if len(uniq) == 0:
            print(f"  {col:<18s} no non-null values")
            continue
        n_fallback = 0
        for smi in uniq:
            mol = smiles_to_mol(smi)
            if mol is None or GetScaffoldForMol(mol).GetNumAtoms() == 0:
                n_fallback += 1
        pct = 100 * n_fallback / len(uniq)
        flag = "  <-- unusable as a grouping key" if pct > 10 else ""
        print(f"  {col:<18s} {n_fallback:>6d}/{len(uniq):<6d} unique SMILES ({pct:5.1f}%) "
              f"have no ring system{flag}")


def report_group_counts(df: pd.DataFrame, keys: Dict[str, KeySpec],
                        mode: str) -> Dict[str, Dict[str, pd.Series]]:
    """Print unique-group counts per key, under one or both scaffold flavours.

    Args:
        df: Source DataFrame.
        keys: Grouping-key mapping.
        mode: "generic", "murcko", or "both".

    Returns:
        Mapping of flavour name to its {key name: Series} mapping, so callers can
        reuse the (expensive) scaffold computation.
    """
    print("\n" + "=" * 86)
    print("3. UNIQUE GROUPS PER KEY")
    print("=" * 86)
    print("Fewer groups = coarser = a stronger constraint on the split.")
    print("Murcko strips terminal attachment dummies ([*:1]); generic keeps them as carbon,")
    print("so on component columns Murcko often yields the coarser (safer) grouping.\n")

    computed: Dict[str, Dict[str, pd.Series]] = {}
    flavours = ["generic", "murcko"] if mode == "both" else [mode]
    for flavour in flavours:
        computed[flavour] = compute_scaffold_keys(df, keys, generic=(flavour == "generic"))

    print(f"  {'key':<22s}" + "".join(f"{f:>12s}" for f in flavours))
    for name in keys:
        row = f"  {name:<22s}"
        for flavour in flavours:
            row += f"{computed[flavour][name].nunique():>12d}"
        print(row)
    return computed


def report_group_sizes(group_keys: Dict[str, pd.Series], n_rows: int, n_folds: int) -> pd.DataFrame:
    """Print the per-key group-size distribution and n_folds feasibility.

    Args:
        group_keys: Mapping from key name to per-row group Series.
        n_rows: Total row count, for the ideal-fold-size comparison.
        n_folds: Number of folds the split must support.

    Returns:
        Tidy DataFrame of the printed statistics, one row per key.
    """
    print("\n" + "=" * 86)
    print(f"4. GROUP SIZES  (ideal fold = {n_rows / n_folds:.0f} rows at {n_folds} folds)")
    print("=" * 86)
    print("A group larger than one fold makes balanced folds impossible by construction.")
    print("Many singletons mean the 'grouping' barely constrains the split at all.\n")
    ideal = n_rows / n_folds
    print(f"  {'key':<22s}{'groups':>8s}{'max':>8s}{'max%':>8s}{'median':>8s}"
          f"{'single':>8s}{'single%':>9s}  verdict")
    rows = []
    for name, series in group_keys.items():
        vc = series.value_counts()
        n_single = int((vc == 1).sum())
        pct_single_rows = 100 * n_single / max(len(series.dropna()), 1)
        if vc.max() >= ideal:
            verdict = "TOO COARSE - cannot balance folds"
        elif pct_single_rows > 40:
            verdict = "too fine - near-random split"
        else:
            verdict = "usable"
        print(f"  {name:<22s}{len(vc):>8d}{vc.max():>8d}{100 * vc.max() / max(n_rows, 1):>7.1f}%"
              f"{vc.median():>8.0f}{n_single:>8d}{pct_single_rows:>8.1f}%  {verdict}")
        rows.append({"key": name, "n_groups": len(vc), "max_group": int(vc.max()),
                     "median_group": float(vc.median()), "n_singletons": n_single,
                     "pct_singleton_rows": pct_single_rows, "verdict": verdict})
    return pd.DataFrame(rows)


def report_leakage(group_keys: Dict[str, pd.Series], base_keys: List[str], n_folds: int,
                   seed: int) -> Optional[pd.DataFrame]:
    """Run the real splitter per candidate key and measure cross-component leakage.

    For each candidate grouping key this builds the actual CV folds via
    `get_fold_indices`, then reports -- averaged over folds -- the fraction of
    held-out rows whose single-column component groups also appear in the training half.

    Args:
        group_keys: Mapping from key name to per-row group Series.
        base_keys: Single-column key names to measure leakage of.
        n_folds: Number of outer folds to simulate.
        seed: Seed passed to the splitter.

    Returns:
        Tidy DataFrame of leakage fractions, or None if the splitter could not be
        imported (it pulls in optuna, absent from the inference environment).
    """
    print("\n" + "=" * 86)
    print(f"5. MEASURED LEAKAGE  (real {n_folds}-fold split, seed {seed})")
    print("=" * 86)
    try:
        from retrotac.models.training import get_fold_indices
    except ImportError as exc:
        print(f"  skipped: could not import get_fold_indices ({exc}).")
        print("  Run inside the training environment/container to enable this report.")
        return None

    print("Fraction of HELD-OUT rows whose component group is also present in TRAIN.")
    print("Lower is better. Note product ('+') keys leak MORE than their parts and")
    print("union ('&') keys drive their constituents to 0.0% -- that is the point.\n")

    work = pd.DataFrame({n: s for n, s in group_keys.items()}).reset_index(drop=True)
    print(f"  {'group by':<22s}" + "".join(f"{'leaks ' + n:>16s}" for n in base_keys))
    rows = []
    for grp in group_keys:
        sim = work.copy()
        sim["scaffolds"] = sim[grp]
        # rows with a missing key cannot be grouped; drop them for this simulation only
        sim = sim.dropna(subset=["scaffolds"]).reset_index(drop=True)
        leak: Dict[str, List[float]] = {other: [] for other in base_keys if other != grp}
        for fold in range(n_folds):
            train_idx, val_idx, _, _ = get_fold_indices(sim, seed, fold, n_folds)
            for other in leak:
                train_vals = set(sim[other].iloc[train_idx].dropna())
                val_vals = sim[other].iloc[val_idx]
                leak[other].append(val_vals.isin(train_vals).sum() / max(len(val_vals), 1))
        line = f"  {grp:<22s}"
        for other in base_keys:
            if other == grp:
                line += f"{'-':>16s}"
            else:
                mean_leak = float(np.mean(leak[other]))
                line += f"{100 * mean_leak:>15.1f}%"
                rows.append({"grouped_by": grp, "leaked_key": other,
                             "frac_val_rows_seen_in_train": mean_leak})
        print(line)
    return pd.DataFrame(rows)


def report_eta_squared(df: pd.DataFrame, group_keys: Dict[str, pd.Series],
                       targets: List[str]) -> Optional[pd.DataFrame]:
    """Print the fraction of each target's variance explained by group membership.

    eta^2 = SS_between / SS_total for the one-way grouping. A value near 1 means the
    label is essentially determined by the group, so that key must define the split.

    Args:
        df: Source DataFrame holding the target columns.
        group_keys: Mapping from key name to per-row group Series.
        targets: Target column names.

    Returns:
        Tidy DataFrame of eta^2 values, or None if no usable targets were given.
    """
    print("\n" + "=" * 86)
    print("6. ETA^2  (fraction of target variance explained by group membership)")
    print("=" * 86)
    if not targets:
        print("  skipped: no usable target columns.")
        return None
    print("1.0 = the label is fully determined by that key. A key that is both")
    print("high-eta^2 here and high-leakage above is the one you should group on.\n")

    # key-major, matching reports 4 and 5, so the candidate keys read down one column
    widths = {t: max(10, len(t) + 2) for t in targets}
    print(f"  {'key':<22s}" + "".join(f"{t:>{widths[t]}s}" for t in targets))
    rows = []
    eta_by_key: Dict[str, Dict[str, float]] = {}
    for target in targets:
        y = pd.to_numeric(df[target], errors="coerce")
        mask = y.notna()
        yy = y[mask]
        usable = len(yy) >= 2 and yy.var() > 0
        grand = yy.mean() if usable else float("nan")
        ss_tot = float(((yy - grand) ** 2).sum()) if usable else 0.0
        for name in group_keys:
            if not usable or ss_tot <= 0:
                eta_by_key.setdefault(name, {})[target] = float("nan")
                continue
            grouped = yy.groupby(group_keys[name][mask])
            ss_between = float((grouped.size() * (grouped.mean() - grand) ** 2).sum())
            eta2 = ss_between / ss_tot
            eta_by_key.setdefault(name, {})[target] = eta2
            rows.append({"target": target, "key": name, "eta_squared": eta2})
    for name in group_keys:
        line = f"  {name:<22s}"
        for target in targets:
            value = eta_by_key[name][target]
            line += (f"{'n/a':>{widths[target]}s}" if np.isnan(value)
                     else f"{value:>{widths[target]}.3f}")
        print(line)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input_csv)
    keys = parse_group_cols(args.group_cols, df)
    keys = add_auto_combos(keys, args.combos, args.combo_keys, args.combo_mode)
    base_keys = [n for n, (mode, _) in keys.items() if mode == "single"]
    targets = select_targets(df, args.targets, keys, args.min_target_rows)

    print(f"Loaded {len(df)} rows from {args.input_csv}")
    print(f"Candidate keys ({len(keys)}): {', '.join(keys)}")
    print(f"Targets ({len(targets)}): {', '.join(targets) if targets else '(none)'}")

    report_overview(df, keys)
    report_acyclic_fallback(df, keys)
    computed = report_group_counts(df, keys, args.scaffold_mode)

    primary = args.primary_mode if args.scaffold_mode == "both" else args.scaffold_mode
    group_keys = computed[primary]
    print(f"\n[reports 4-6 use the '{primary}' scaffold flavour]")

    group_stats = report_group_sizes(group_keys, len(df), args.n_folds)
    leakage = report_leakage(group_keys, base_keys, args.n_folds, args.seed)
    eta = report_eta_squared(df, group_keys, targets)

    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        group_stats.to_csv(out_dir / "group_stats.csv", index=False)
        written = ["group_stats.csv"]
        if leakage is not None:
            leakage.to_csv(out_dir / "leakage.csv", index=False)
            written.append("leakage.csv")
        if eta is not None:
            eta.to_csv(out_dir / "eta_squared.csv", index=False)
            written.append("eta_squared.csv")
        print(f"\nSaved → {out_dir}/{{{', '.join(written)}}}")


if __name__ == "__main__":
    main()
