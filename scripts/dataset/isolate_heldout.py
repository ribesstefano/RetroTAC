"""
isolate_heldout.py
───────────────────
Splits a scored-routes CSV into a train/val set and a diverse held-out
(test) set, using Morgan fingerprints to select the held-out rows via one
of three RDKit-based methods:
    butina   - Butina clustering (distance threshold --butina-cutoff): whole
               clusters are shuffled and greedily assigned to the held-out
               set until closest to --heldout-pct, so near-duplicate
               molecules (within the cutoff) never end up split across
               train/test. The achieved size only approximates --heldout-pct
               since clusters are never split.
    maxmin   - MaxMinPicker: the *most chemically diverse* subset of the
               input hits --heldout-pct exactly, but a picked molecule's
               near-duplicate can remain in train_val.
    adaptive - Sweeps Butina cutoffs over [--adaptive-min-cutoff,
               --adaptive-max-cutoff] (step --adaptive-step), scores each
               clustering via evaluate_clusters (silhouette from precomputed
               Tanimoto distance, plus Davies-Bouldin, Calinski-Harabasz,
               avg-cluster/data-size ratio, and cluster-size skewness), and
               picks the cutoff with the best composite quality score among
               those whose achieved held-out size stays within
               --adaptive-pct-tolerance points of --heldout-pct -- i.e. a
               "sweet spot" cutoff instead of a fixed --butina-cutoff guess.

Also reports whether the target column's distribution is similar between
the two resulting sets (descriptive stats + a two-sample statistical test),
and can optionally render a handful of example molecules from each set.

Usage
-----
    python isolate_heldout.py data/routes/routes_scored.csv \
        --smiles-col smiles --target-col synthesizability \
        --heldout-pct 10 --method butina --butina-cutoff 0.7 \
        --output-dir data/routes --make-figures

Arguments:
    input_csv            Input CSV path.
    --smiles-col          SMILES column name (default: "smiles").
    --target-col          Target column name (default: "synthesizability").
    --heldout-pct         Percentage (0-100) of rows to isolate into the
                          held-out set (default: 10.0).
    --method              Diversity-selection method: "maxmin", "butina", or
                          "adaptive" (default: "butina").
    --butina-cutoff       Tanimoto distance threshold for Butina clustering;
                          only used with --method butina (default: 0.7).
    --adaptive-min-cutoff Smallest cutoff swept with --method adaptive
                          (default: 0.5).
    --adaptive-max-cutoff Largest cutoff swept with --method adaptive
                          (default: 0.95).
    --adaptive-step       Step between swept cutoffs (default: 0.05).
    --adaptive-pct-tolerance
                          Max deviation (pct points) from --heldout-pct
                          allowed when picking the best cutoff (default: 3.0).
    --output-dir          Output directory (default: "data/routes").
    --train-val-filename  Filename for the non-held-out split
                          (default: "routes_train_val.csv").
    --test-filename       Filename for the held-out split
                          (default: "routes_test.csv").
    --seed                Random seed for the picker/cluster-order and
                          figure sampling (default: 42).
    --make-figures        If set, also render up to --n-figure-mols example
                          molecules from each set as grid images, plus an
                          overlapping train_val-vs-test target distribution
                          plot, under <output-dir>/smiles_figures/.
    --n-figure-mols       Max molecules per figure (default: 20).
"""

import argparse
from array import array
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import DataStructs, RDLogger
from rdkit.Chem import Draw
from rdkit.ML.Cluster import Butina
from rdkit.SimDivFilters import rdSimDivPickers
from scipy import stats
from scipy.spatial.distance import squareform
from tabulate import tabulate
from tqdm import tqdm
from scipy.stats import skew
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

from protac_synth.chem_utils import morgan_fp, papply, smiles_to_mol

RDLogger.DisableLog("rdApp.*")

MAX_CATEGORICAL_UNIQUE = 10

def evaluate_clusters(
    X: np.ndarray, clusters: np.ndarray, dist_matrix: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """Compute cluster-quality metrics and cluster-size statistics for a clustering.

    Args:
        X: Feature matrix, shape [n_samples, n_features] (e.g. a dense fingerprint
            bit-matrix), row-aligned with *clusters*. Used for Davies-Bouldin and
            Calinski-Harabasz (sklearn has no precomputed-distance option for either),
            and as the silhouette input when *dist_matrix* is omitted.
        clusters: Per-sample cluster-label array, shape [n_samples].
        dist_matrix: Optional square pairwise-distance matrix, shape [n_samples, n_samples]
            (e.g. Tanimoto distance). When given, silhouette is computed from this via
            ``metric="precomputed"`` instead of Euclidean distance on *X* -- Euclidean
            distance on binary fingerprint bits doesn't track chemical similarity, and
            systematically biases silhouette toward finer clusterings regardless of
            whether they're chemically meaningful.

    Returns:
        Dict with sklearn cluster-quality scores (``silhouette``, ``davies_bouldin``,
        ``calinski_harabasz``) and cluster-size statistics (``avg_cluster_size``,
        ``avg_cluster_data_ratio``, ``std_cluster_size``, ``min_cluster_size``,
        ``median_cluster_size``, ``max_cluster_size``, ``cluster_size_skewness``,
        ``num_clusters``). A degenerate single-cluster input returns sentinel scores
        (-1 / inf) for the sklearn metrics instead of raising, since those all require
        at least 2 clusters.
    """
    unique_clusters = list(set(clusters))

    if len(unique_clusters) < 2:  # Avoid single-cluster issues
        return {
            "silhouette": -1,
            "davies_bouldin": float("inf"),
            "calinski_harabasz": -1,
            "avg_cluster_size": len(X),
            "avg_cluster_data_ratio": 1,
            "std_cluster_size": 0,
            "min_cluster_size": len(X),
            "median_cluster_size": len(X),
            "max_cluster_size": len(X),
            "cluster_size_skewness": 0,
            "num_clusters": 1,
        }

    # Compute standard clustering metrics
    if dist_matrix is not None:
        silhouette = silhouette_score(dist_matrix, clusters, metric="precomputed")
    else:
        silhouette = silhouette_score(X, clusters)
    davies_bouldin = davies_bouldin_score(X, clusters)
    calinski_harabasz = calinski_harabasz_score(X, clusters)

    # Compute cluster size statistics
    cluster_sizes = [len(np.where(clusters == i)[0]) for i in np.unique(clusters)]
    avg_cluster_size = np.mean(cluster_sizes)
    avg_cluster_data_ratio = avg_cluster_size / len(X)
    std_cluster_size = np.std(cluster_sizes)
    median_cluster_size = np.median(cluster_sizes)
    min_cluster_size = np.min(cluster_sizes)
    max_cluster_size = np.max(cluster_sizes)
    cluster_size_skewness = skew(cluster_sizes, nan_policy="omit")  # Indicates imbalance in cluster sizes

    return {
        "silhouette": silhouette,
        "davies_bouldin": davies_bouldin,
        "calinski_harabasz": calinski_harabasz,
        "avg_cluster_size": avg_cluster_size,
        "avg_cluster_data_ratio": avg_cluster_data_ratio,
        "std_cluster_size": std_cluster_size,
        "min_cluster_size": min_cluster_size,
        "median_cluster_size": median_cluster_size,
        "max_cluster_size": max_cluster_size,
        "cluster_size_skewness": cluster_size_skewness,
        "num_clusters": len(unique_clusters),
    }

def _is_categorical(series: pd.Series) -> bool:
    """Heuristic: non-numeric dtype, or few enough unique values to be a class label."""
    return series.dtype == object or series.nunique() <= MAX_CATEGORICAL_UNIQUE


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input_csv", type=str, help="Input CSV path.")
    p.add_argument("--smiles-col", type=str, default="smiles", help="SMILES column name.")
    p.add_argument("--target-col", type=str, default="synthesizability", help="Target column name.")
    p.add_argument(
        "--heldout-pct",
        type=float,
        default=10.0,
        help="Percentage (0-100) of rows to isolate into the held-out set (default: 10.0).",
    )
    p.add_argument(
        "--method",
        type=str,
        choices=["maxmin", "butina", "adaptive"],
        default="butina",
        help="Diversity-selection method for the held-out set (default: butina).",
    )
    p.add_argument(
        "--butina-cutoff",
        type=float,
        default=0.7,
        help="Tanimoto distance threshold for Butina clustering; only used with --method butina (default: 0.7).",
    )
    p.add_argument(
        "--adaptive-min-cutoff",
        type=float,
        default=0.5,
        help="Smallest Butina cutoff to sweep with --method adaptive (default: 0.5).",
    )
    p.add_argument(
        "--adaptive-max-cutoff",
        type=float,
        default=0.95,
        help="Largest Butina cutoff to sweep with --method adaptive (default: 0.95).",
    )
    p.add_argument(
        "--adaptive-step",
        type=float,
        default=0.05,
        help="Step between swept cutoffs with --method adaptive (default: 0.05).",
    )
    p.add_argument(
        "--adaptive-pct-tolerance",
        type=float,
        default=3.0,
        help="Max deviation, in percentage points, from --heldout-pct allowed when picking the "
        "best cutoff with --method adaptive (default: 3.0).",
    )
    p.add_argument("--output-dir", type=str, default="data/routes", help="Output directory.")
    p.add_argument(
        "--train-val-filename", type=str, default="routes_train_val.csv", help="Non-held-out split filename."
    )
    p.add_argument("--test-filename", type=str, default="routes_test.csv", help="Held-out split filename.")
    p.add_argument(
        "--seed", type=int, default=42, help="Random seed for the picker/cluster-order and figure sampling."
    )
    p.add_argument(
        "--make-figures",
        action="store_true",
        help="Render up to --n-figure-mols example molecules per set under <output-dir>/smiles_figures/.",
    )
    p.add_argument("--n-figure-mols", type=int, default=20, help="Max molecules per figure (default: 20).")
    return p.parse_args()


def pick_heldout_indices_maxmin(fps: List, pct: float, seed: int) -> List[int]:
    """Select the most chemically diverse subset of fingerprints via MaxMinPicker.

    Args:
        fps: RDKit ``ExplicitBitVect`` Morgan fingerprints, one per row.
        pct: Percentage (0-100) of rows to select.
        seed: Random seed for the picker's first choice.

    Returns:
        Row indices (into *fps*) of the selected, maximally diverse subset.
    """
    n = len(fps)
    pick_size = round(n * pct / 100.0)
    if pick_size < 1 or pick_size >= n:
        raise ValueError(f"--heldout-pct={pct} yields pick_size={pick_size} for n={n}; must be in [1, {n - 1}].")
    picker = rdSimDivPickers.MaxMinPicker()
    picked = picker.LazyBitVectorPick(fps, n, pick_size, seed=seed)
    return list(picked)


def _pairwise_tanimoto_condensed(fps: List) -> array:
    """Compute the condensed (lower-triangle) pairwise Tanimoto-distance list.

    This is the flat format RDKit's ``Butina.ClusterData`` expects as ``data``
    when ``isDistData=True``.

    Args:
        fps: RDKit ``ExplicitBitVect`` Morgan fingerprints, one per row.

    Returns:
        ``array('f')`` of length n*(n-1)/2 holding 1 - Tanimoto similarity for
        each pair, in row-major lower-triangle order.
    """
    n = len(fps)
    dists = array("f")
    for i in tqdm(range(1, n), desc="Pairwise Tanimoto distances"):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend(1.0 - s for s in sims)
    return dists


def _clusters_to_labels(clusters: Tuple[Tuple[int, ...], ...], n: int) -> np.ndarray:
    """Convert Butina's tuple-of-index-tuples clusters into a per-row label array.

    Args:
        clusters: Output of ``Butina.ClusterData``: a tuple of tuples of row indices.
        n: Total number of rows (length of the returned array).

    Returns:
        Integer array of shape [n]; ``labels[i]`` is the cluster index containing row i.
    """
    labels = np.empty(n, dtype=int)
    for cluster_id, idxs in enumerate(clusters):
        labels[list(idxs)] = cluster_id
    return labels


def _fill_clusters_to_target(clusters: Tuple[Tuple[int, ...], ...], target: int, seed: int) -> List[int]:
    """Greedily assign whole, seed-shuffled clusters to a selection until *target* size is reached.

    Args:
        clusters: Output of ``Butina.ClusterData``: a tuple of tuples of row indices.
        target: Desired selection size, in rows.
        seed: Random seed for shuffling cluster order.

    Returns:
        Row indices of the selected rows (whole clusters only); size is >= target
        unless every cluster was exhausted first.
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(clusters))
    selected: List[int] = []
    for i in order:
        selected.extend(clusters[i])
        if len(selected) >= target:
            break
    return selected


def _fps_to_dense(fps: List) -> np.ndarray:
    """Convert a list of RDKit ``ExplicitBitVect`` fingerprints into a dense bit-matrix.

    Args:
        fps: RDKit ``ExplicitBitVect`` fingerprints, one per row (all the same length).

    Returns:
        Array of shape [len(fps), n_bits], values in {0, 1}.
    """
    n_bits = fps[0].GetNumBits()
    out = np.zeros((len(fps), n_bits), dtype=np.uint8)
    tmp = np.zeros(n_bits, dtype=np.uint8)
    for i, fp in enumerate(fps):
        DataStructs.ConvertToNumpyArray(fp, tmp)
        out[i] = tmp
    return out


def pick_heldout_indices_butina(fps: List, pct: float, cutoff: float, seed: int) -> List[int]:
    """Select a held-out set as whole Butina clusters, targeting *pct* of rows.

    Clusters all molecules at Tanimoto distance <= *cutoff* (rdkit.ML.Cluster.Butina),
    then shuffles the resulting clusters (seeded) and greedily assigns whole
    clusters to the held-out set until its size reaches *pct*. Clusters are never
    split, so near-duplicate molecules never end up on both sides of the split --
    the achieved size only approximates *pct* as a result.

    Args:
        fps: RDKit ``ExplicitBitVect`` Morgan fingerprints, one per row.
        pct: Target percentage (0-100) of rows to select.
        cutoff: Tanimoto distance threshold for cluster membership.
        seed: Random seed for shuffling cluster order.

    Returns:
        Row indices (into *fps*) of the selected rows (whole clusters only).
    """
    n = len(fps)
    # Whole-cluster assignment below only approximates this.
    target = round(n * pct / 100.0)
    if target < 1 or target >= n:
        raise ValueError(f"--heldout-pct={pct} yields target={target} for n={n}; must be in [1, {n - 1}].")

    # O(n^2) and the dominant cost; clustering itself is cheap after this.
    dists = _pairwise_tanimoto_condensed(fps)
    clusters = Butina.ClusterData(dists, n, cutoff, isDistData=True)
    print(f"  Butina clustering (cutoff={cutoff}) produced {len(clusters)} clusters from {n} molecules.")

    # Whole clusters only, so near-duplicates never split across train/test.
    selected = _fill_clusters_to_target(clusters, target, seed)
    print(f"  Achieved held-out size: {len(selected)} rows ({100.0 * len(selected) / n:.2f}%, target was {pct}%).")
    return selected


def pick_heldout_indices_adaptive(
    fps: List,
    X: np.ndarray,
    pct: float,
    seed: int,
    min_cutoff: float,
    max_cutoff: float,
    step: float,
    pct_tolerance: float,
) -> Tuple[List[int], float, pd.DataFrame]:
    """Sweep Butina cutoffs and pick the one balancing cluster quality against --heldout-pct.

    Reclusters at every cutoff in ``[min_cutoff, max_cutoff]`` (step *step*) from a single
    precomputed pairwise Tanimoto-distance matrix, scores each clustering with
    ``evaluate_clusters`` (silhouette computed from that same Tanimoto distance matrix via
    ``metric="precomputed"``, not Euclidean distance on raw fingerprint bits), and restricts
    candidates to those whose whole-cluster-filled held-out size lands within *pct_tolerance*
    percentage points of *pct* (falling back to the single closest-achieved cutoff if none
    qualify). Among the survivors, the cutoff with the highest composite quality score -- the
    mean of min-max-normalized silhouette, negated Davies-Bouldin, Calinski-Harabasz, negated
    avg-cluster/data-size ratio (penalizes one dominant cluster), and negated absolute
    cluster-size skewness (penalizes a lopsided size distribution) -- is selected as the
    "sweet spot".

    Args:
        fps: RDKit ``ExplicitBitVect`` Morgan fingerprints, one per row.
        X: Dense fingerprint bit-matrix, shape [n_samples, n_bits], for cluster-quality scoring.
        pct: Target percentage (0-100) of rows to select.
        seed: Random seed for shuffling cluster order at every cutoff.
        min_cutoff: Smallest Tanimoto distance threshold to try.
        max_cutoff: Largest Tanimoto distance threshold to try.
        step: Increment between swept thresholds.
        pct_tolerance: Max allowed deviation (percentage points) from *pct*.

    Returns:
        ``(selected_indices, chosen_cutoff, metrics_df)``: the held-out row indices at the
        chosen cutoff, that cutoff, and a DataFrame with one row per swept cutoff (metrics
        plus the composite ``quality_score``), for logging/plotting.
    """
    n = len(fps)
    target = round(n * pct / 100.0)
    if target < 1 or target >= n:
        raise ValueError(f"--heldout-pct={pct} yields target={target} for n={n}; must be in [1, {n - 1}].")

    # O(n^2) and computed once; reclustering per cutoff from this is cheap.
    dists = _pairwise_tanimoto_condensed(fps)
    # Full matrix, built once, for silhouette's metric="precomputed" (needs random
    # access, not just the lower triangle).
    dist_matrix = squareform(np.asarray(dists, dtype=np.float32))

    rows = []
    selections: Dict[float, List[int]] = {}
    for raw_cutoff in np.arange(min_cutoff, max_cutoff + 1e-9, step):
        cutoff = round(float(raw_cutoff), 4)
        clusters = Butina.ClusterData(dists, n, cutoff, isDistData=True)
        labels = _clusters_to_labels(clusters, n)
        # Tanimoto silhouette via dist_matrix; X only feeds DB/CH (no precomputed option there).
        metrics = evaluate_clusters(X, labels, dist_matrix=dist_matrix)
        # Same greedy fill as pick_heldout_indices_butina, for a comparable achieved_pct.
        selected = _fill_clusters_to_target(clusters, target, seed)
        achieved_pct = 100.0 * len(selected) / n
        selections[cutoff] = selected
        rows.append({"cutoff": cutoff, "achieved_pct": achieved_pct, **metrics})
        print(
            f"  cutoff={cutoff:.3f}: {metrics['num_clusters']} clusters, "
            f"silhouette={metrics['silhouette']:.4f}, achieved={achieved_pct:.2f}%"
        )

    metrics_df = pd.DataFrame(rows)

    # Min-max to [0, 1]; sign-flipped first so "lower is better" also means higher = better.
    def _norm(s: pd.Series, higher_is_better: bool) -> pd.Series:
        s = s if higher_is_better else -s
        spread = s.max() - s.min()
        return (s - s.min()) / spread if spread > 0 else pd.Series(0.5, index=s.index)

    # A 1-cluster collapse's sentinel scores (silhouette=-1, davies_bouldin=inf) would
    # poison normalization for every cutoff, so exclude those from scoring/selection.
    valid = metrics_df["num_clusters"] >= 2
    if not valid.any():
        raise ValueError(
            "Every swept cutoff collapsed to a single cluster; "
            "narrow --adaptive-min-cutoff/--adaptive-max-cutoff."
        )

    # Rewards tight/separated clusters, penalizes a lopsided split (dominant cluster or skewed sizes).
    metrics_df["quality_score"] = np.nan
    metrics_df.loc[valid, "quality_score"] = (
        _norm(metrics_df.loc[valid, "silhouette"], higher_is_better=True)
        + _norm(metrics_df.loc[valid, "davies_bouldin"], higher_is_better=False)
        + _norm(metrics_df.loc[valid, "calinski_harabasz"], higher_is_better=True)
        + _norm(metrics_df.loc[valid, "avg_cluster_data_ratio"], higher_is_better=False)
        + _norm(metrics_df.loc[valid, "cluster_size_skewness"].abs(), higher_is_better=False)
    ) / 5.0

    # Only cutoffs whose achieved size is actually usable -- quality doesn't matter otherwise.
    candidates = metrics_df[valid]
    within_tol = candidates[(candidates["achieved_pct"] - pct).abs() <= pct_tolerance]
    if within_tol.empty:
        print(f"  No cutoff landed within +/-{pct_tolerance} pts of {pct}%; using the closest achieved size.")
        within_tol = candidates.loc[[(candidates["achieved_pct"] - pct).abs().idxmin()]]

    # Best composite quality among the size-acceptable candidates.
    best_row = within_tol.loc[within_tol["quality_score"].idxmax()]
    chosen_cutoff = best_row["cutoff"]
    print(
        f"  Chosen cutoff={chosen_cutoff:.3f}: quality_score={best_row['quality_score']:.4f}, "
        f"achieved={best_row['achieved_pct']:.2f}% (target {pct}%)."
    )
    return selections[chosen_cutoff], chosen_cutoff, metrics_df


def describe_distribution(train_val: pd.Series, test: pd.Series, target_col: str) -> str:
    """Compare the target column's distribution between the two splits.

    Auto-detects continuous vs. categorical (<= MAX_CATEGORICAL_UNIQUE unique
    values, or non-numeric dtype) and reports descriptive stats plus a
    two-sample statistical test (Kolmogorov-Smirnov for continuous,
    chi-square for categorical).

    Args:
        train_val: Target column values for the non-held-out split.
        test: Target column values for the held-out split.
        target_col: Column name, used only for the report header.

    Returns:
        Human-readable report string.
    """
    # Drop NaNs (e.g. unresolved routes) so they don't skew the stats/test below.
    train_val = train_val.dropna()
    test = test.dropna()

    lines = [f"=== Target column '{target_col}' distribution: train_val vs. held-out test ==="]
    if _is_categorical(train_val):
        # Per-class counts, aligned on the union of classes across both splits.
        counts = pd.concat(
            [train_val.value_counts().rename("train_val"), test.value_counts().rename("test")], axis=1
        ).fillna(0)
        # Normalized per split, not by grand total, so proportions are comparable despite the size difference.
        fracs = counts / counts.sum(axis=0)
        table = pd.concat(
            [counts.astype(int).add_suffix("_count"), fracs.round(4).add_suffix("_frac")], axis=1
        ).sort_index()
        lines.append(tabulate(table, headers="keys", tablefmt="github"))
        # Tests whether the class mix actually differs between splits.
        chi2, p_value, _, _ = stats.chi2_contingency(counts)
        lines.append(f"\nChi-square test of independence: statistic={chi2:.4f}, p-value={p_value:.4g}")
    else:
        summary = pd.DataFrame({"train_val": train_val.describe(), "test": test.describe()})
        lines.append(tabulate(summary, headers="keys", tablefmt="github"))
        # Tests whether both splits could plausibly share the same distribution.
        ks_stat, p_value = stats.ks_2samp(train_val, test)
        lines.append(f"\nTwo-sample Kolmogorov-Smirnov test: statistic={ks_stat:.4f}, p-value={p_value:.4g}")

    # Same alpha for either branch, so the verdict line reads consistently.
    verdict = "similar" if p_value >= 0.05 else "significantly different"
    lines.append(f"-> p-value {'>=' if p_value >= 0.05 else '<'} 0.05: distributions look {verdict} (alpha=0.05).")
    return "\n".join(lines)


def plot_target_overlap(train_val: pd.Series, test: pd.Series, target_col: str, out_path: Path) -> None:
    """Save a plot overlapping the target column's distribution between the two splits.

    Continuous targets get overlaid, density-normalized histograms (raw counts
    would be misleading since the two splits have very different sizes);
    categorical targets get a grouped bar chart of per-class fractions.

    Args:
        train_val: Target column values for the non-held-out split.
        test: Target column values for the held-out split.
        target_col: Column name, used for axis/title labels.
        out_path: PNG file path to write (parent dir created if missing).
    """
    train_val = train_val.dropna()
    test = test.dropna()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 5))
    if _is_categorical(train_val):
        categories = sorted(set(train_val) | set(test))
        x = np.arange(len(categories))
        width = 0.35
        train_frac = train_val.value_counts(normalize=True).reindex(categories, fill_value=0)
        test_frac = test.value_counts(normalize=True).reindex(categories, fill_value=0)
        ax.bar(x - width / 2, train_frac, width, alpha=0.7, label="train_val")
        ax.bar(x + width / 2, test_frac, width, alpha=0.7, label="test (held-out)")
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.set_ylabel("Fraction")
    else:
        bins = np.histogram_bin_edges(pd.concat([train_val, test]), bins=30)
        ax.hist(train_val, bins=bins, alpha=0.5, density=True, label="train_val")
        ax.hist(test, bins=bins, alpha=0.5, density=True, label="test (held-out)")
        ax.set_ylabel("Density")

    ax.set_xlabel(target_col)
    ax.set_title(f"'{target_col}' distribution: train_val vs. held-out test")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_adaptive_metrics(metrics_df: pd.DataFrame, chosen_cutoff: float, out_path: Path) -> None:
    """Save a diagnostic plot of cluster-quality metrics vs. swept Butina cutoff.

    Silhouette (computed from precomputed Tanimoto distance), (negated) Davies-Bouldin,
    and Calinski-Harabasz are min-max normalized to [0, 1] before plotting (their raw
    scales aren't comparable), alongside the composite ``quality_score`` (which also
    folds in the avg-cluster/data-size ratio and cluster-size skewness -- not separately
    plotted here to keep the legend to 3x2). Achieved held-out percentage is shown on a
    secondary axis since it lives on a 0-100 scale.

    Cutoffs that collapsed to a single cluster (``num_clusters < 2``, no ``quality_score``)
    are excluded from the normalized-metric lines but still show up on the achieved-%
    line, so a degenerate high cutoff is visible without corrupting the other curves.

    Args:
        metrics_df: Per-cutoff metrics from ``pick_heldout_indices_adaptive``.
        chosen_cutoff: The selected cutoff, marked with a vertical line.
        out_path: PNG file path to write (parent dir created if missing).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    valid = metrics_df[metrics_df["quality_score"].notna()]

    def _norm01(s: pd.Series, higher_is_better: bool) -> pd.Series:
        s = s if higher_is_better else -s
        spread = s.max() - s.min()
        return (s - s.min()) / spread if spread > 0 else pd.Series(0.5, index=s.index)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(valid["cutoff"], _norm01(valid["silhouette"], True), marker="o", label="Silhouette (norm.)")
    ax.plot(
        valid["cutoff"], _norm01(valid["davies_bouldin"], False), marker="o",
        label="Davies-Bouldin (norm., inverted)",
    )
    ax.plot(
        valid["cutoff"], _norm01(valid["calinski_harabasz"], True), marker="o",
        label="Calinski-Harabasz (norm.)",
    )
    ax.plot(valid["cutoff"], valid["quality_score"], marker="o", linewidth=2, color="black",
            label="Quality score (mean)")
    ax.axvline(chosen_cutoff, color="gray", linestyle="--", alpha=0.7, label=f"Chosen cutoff ({chosen_cutoff:.2f})")
    ax.set_xlabel("Butina cutoff (Tanimoto distance)")
    ax.set_ylabel("Normalized metric value")

    ax2 = ax.twinx()
    ax2.plot(metrics_df["cutoff"], metrics_df["achieved_pct"], marker="s", linestyle=":", color="tab:red",
              label="Achieved held-out %")
    ax2.set_ylabel("Achieved held-out %", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(
        lines1 + lines2, labels1 + labels2,
        loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize="small",
    )
    ax.set_title("Cluster-quality metrics vs. Butina cutoff")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_figures(
    df: pd.DataFrame, smiles_col: str, target_col: str, out_dir: Path, n: int, seed: int, name: str
) -> None:
    """Render a grid image of up to *n* sampled molecules from *df*.

    Args:
        df: DataFrame with a ``_mol`` column of pre-parsed RDKit Mols.
        smiles_col: SMILES column name, used only for the sub-image legend fallback.
        target_col: Target column name, shown as the sub-image legend.
        out_dir: Directory the PNG is written into (created if missing).
        n: Max number of molecules to sample and render.
        seed: Random seed for sampling.
        name: Output filename (without directory).
    """
    sample = df.sample(n=min(n, len(df)), random_state=seed)
    legends = [f"{v:.3f}" if pd.notna(v) else "NA" for v in sample[target_col]]
    img = Draw.MolsToGridImage(
        sample["_mol"].tolist(), molsPerRow=5, subImgSize=(250, 250), legends=legends
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    img.save(out_dir / name)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input_csv)
    # Fail fast on a typo'd column name, not minutes into fingerprinting.
    for col in (args.smiles_col, args.target_col):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in {args.input_csv}. Available: {df.columns.tolist()}")

    print(f"Loaded {len(df)} rows from {args.input_csv}")
    # Parsed once; every downstream step reuses these Mols.
    df["_mol"] = papply(df[args.smiles_col], smiles_to_mol, desc="Parsing SMILES")
    n_invalid = df["_mol"].isna().sum()
    if n_invalid:
        # Drop unparseable rows before the row count (used for --heldout-pct) is fixed.
        print(f"Dropping {n_invalid} rows with unparseable SMILES")
        df = df[df["_mol"].notna()].reset_index(drop=True)

    # Drives every selection method below, each consuming it differently.
    df["_fp"] = papply(df["_mol"], morgan_fp, desc="Computing Morgan fingerprints")

    print(f"Picking the held-out set via {args.method} (target {args.heldout_pct}%, seed={args.seed}) …")
    # Only --method adaptive populates these; gates the diagnostic CSV/plot below.
    metrics_df = None
    chosen_cutoff = None
    if args.method == "maxmin":
        heldout_idx = pick_heldout_indices_maxmin(df["_fp"].tolist(), args.heldout_pct, args.seed)
    elif args.method == "butina":
        heldout_idx = pick_heldout_indices_butina(
            df["_fp"].tolist(), args.heldout_pct, args.butina_cutoff, args.seed
        )
    else:
        # Only adaptive needs this dense bit-matrix, for Davies-Bouldin/Calinski-Harabasz.
        X = _fps_to_dense(df["_fp"].tolist())
        heldout_idx, chosen_cutoff, metrics_df = pick_heldout_indices_adaptive(
            df["_fp"].tolist(), X, args.heldout_pct, args.seed,
            args.adaptive_min_cutoff, args.adaptive_max_cutoff, args.adaptive_step,
            args.adaptive_pct_tolerance,
        )
        metrics_path = out_dir / "adaptive_cluster_metrics.csv"
        metrics_df.to_csv(metrics_path, index=False)
        print(f"Saved → {metrics_path}")

    # Boolean mask, not just indices -- both it and its complement are needed below.
    is_heldout = np.zeros(len(df), dtype=bool)
    is_heldout[heldout_idx] = True

    test_df = df[is_heldout].drop(columns=["_mol", "_fp"])
    train_val_df = df[~is_heldout].drop(columns=["_mol", "_fp"])
    print(f"train_val: {len(train_val_df)} rows | test (held-out): {len(test_df)} rows")

    train_val_path = out_dir / args.train_val_filename
    test_path = out_dir / args.test_filename
    train_val_df.to_csv(train_val_path, index=False)
    test_df.to_csv(test_path, index=False)
    print(f"Saved → {train_val_path}")
    print(f"Saved → {test_path}")

    # Sanity-check: a diverse split is only useful if the target distribution isn't wildly different.
    report = describe_distribution(train_val_df[args.target_col], test_df[args.target_col], args.target_col)
    print("\n" + report)
    report_path = out_dir / "heldout_split_report.txt"
    report_path.write_text(report + "\n")
    print(f"\nSaved → {report_path}")

    # Diagnostic/visual only; the CSVs above are already written regardless.
    if args.make_figures:
        fig_dir = out_dir / "smiles_figures"
        save_figures(
            df[~is_heldout], args.smiles_col, args.target_col, fig_dir, args.n_figure_mols, args.seed,
            "train_val_sample.png",
        )
        save_figures(
            df[is_heldout], args.smiles_col, args.target_col, fig_dir, args.n_figure_mols, args.seed,
            "test_heldout_sample.png",
        )
        plot_target_overlap(
            train_val_df[args.target_col], test_df[args.target_col], args.target_col,
            fig_dir / "target_distribution_overlap.png",
        )
        if metrics_df is not None:
            # Only --method adaptive populates metrics_df; maxmin/butina skip this plot.
            plot_adaptive_metrics(metrics_df, chosen_cutoff, fig_dir / "adaptive_metrics_vs_threshold.png")
        print(f"Saved figures → {fig_dir}")


if __name__ == "__main__":
    main()
