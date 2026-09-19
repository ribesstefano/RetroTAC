"""
chem_utils.py
=============
Shared RDKit chemistry utilities used across data preprocessing, diversity
selection, and scoring scripts.

Callers outside ``src/`` must add ``src/`` to ``sys.path`` before importing.
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol, MakeScaffoldGeneric
from rdkit.ML.Descriptors import MoleculeDescriptors
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from retrotac.descriptor_names import DESCRIPTOR_NAMES

RDLogger.DisableLog("rdApp.*")

logger = logging.getLogger(__name__)


# ── SMILES primitives ─────────────────────────────────────────────────────────


def smiles_to_mol(smi: Any) -> Optional[Any]:
    """Parse a SMILES string into an RDKit Mol, handling NaN and non-string inputs.

    Args:
        smi: Input value; may be a string, float NaN, or None.

    Returns:
        RDKit Mol, or None if the input is empty, NaN, or unparseable.
    """
    if pd.isna(smi):
        return None
    s = str(smi).strip()
    return Chem.MolFromSmiles(s) if s else None


def canon_smiles(smi: Any) -> Optional[str]:
    """Return the RDKit canonical SMILES for *smi*, or None if unparseable.

    Args:
        smi: SMILES string (or any value accepted by ``smiles_to_mol``).

    Returns:
        Canonical SMILES string, or None.
    """
    mol = smiles_to_mol(smi)
    return Chem.MolToSmiles(mol) if mol else None


def std_smiles(smi: Any) -> Optional[str]:
    """Standardize a SMILES string: keep largest fragment, neutralize, canonicalize.

    Uses ``sanitize=False`` on initial parse so that unusual valence / aromaticity
    patterns do not raise before the sanitization step, which gives RDKit a chance
    to clean them up.

    Args:
        smi: SMILES string (or any value accepted by ``smiles_to_mol``).

    Returns:
        Canonical isomeric SMILES after standardization, or None on failure.
    """
    if pd.isna(smi):
        return None
    s = str(smi).strip()
    if not s:
        return None
    mol = Chem.MolFromSmiles(s, sanitize=False)
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
        mol = rdMolStandardize.FragmentParent(mol)
        mol = rdMolStandardize.Uncharger().uncharge(mol)
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except Exception:
        return None


def standardize_all(smiles_list: List[Any]) -> List[Optional[Any]]:
    """Standardize a batch of SMILES via ``std_smiles``, returning parsed Mols.

    Downstream feature computation (``compute_fingerprints``, ``compute_descriptors``)
    consumes Mols so each molecule is only parsed/standardized once.

    Args:
        smiles_list: Raw SMILES strings.

    Returns:
        List of RDKit Mol (or None where standardization/parsing failed),
        aligned with *smiles_list*.
    """
    return [smiles_to_mol(std_smiles(s)) for s in smiles_list]


def smiles_hash(smiles: str, length: int = 8) -> str:
    """Return the first *length* hex characters of the SHA-256 hash of *smiles*.

    Used to generate stable, order-independent component IDs (e.g. ``WH_a1b2c3d4``).

    Args:
        smiles: Canonical SMILES string to hash.
        length: Number of hex characters to keep (default 8).

    Returns:
        Truncated hex string of exactly *length* characters.
    """
    return hashlib.sha256(smiles.encode()).hexdigest()[:length]


# ── Fingerprinting ────────────────────────────────────────────────────────────


def morgan_fp(mol: Any, radius: int = 8, nbits: int = 2048) -> Any:
    """Compute a Morgan (circular) fingerprint for a RDKit Mol.

    Args:
        mol: RDKit Mol object.
        radius: Bond-hop radius from each heavy atom center.
        nbits: Bit vector length.

    Returns:
        RDKit ``ExplicitBitVect`` fingerprint.
    """
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nbits)
    return gen.GetFingerprint(mol)


def tanimoto_distance(fp1: Any, fp2: Any) -> float:
    """Return the Tanimoto distance (1 − similarity) between two fingerprints.

    Args:
        fp1: First RDKit fingerprint.
        fp2: Second RDKit fingerprint.

    Returns:
        Distance in [0, 1]; 0 means identical, 1 means disjoint.
    """
    return 1.0 - DataStructs.TanimotoSimilarity(fp1, fp2)


def smiles_to_component_fp(smi: Any) -> Optional[Any]:
    """Compute a Morgan fingerprint for a PROTAC component SMILES with open attachment points.

    Replaces ``[*:n]`` and bare ``*`` dummy atoms with ``[H]`` so that fragments
    with open valences can be fingerprinted without RDKit parse errors.

    Args:
        smi: Component SMILES, potentially containing ``[*:1]``, ``[*:2]``, or ``*``.

    Returns:
        Morgan fingerprint after dummy-atom substitution and H removal,
        or None if the SMILES is empty or unparseable.
    """
    if pd.isna(smi):
        return None
    s = str(smi).strip()
    if not s:
        return None
    s = re.sub(r"\[\*:\d+\]", "[H]", s)
    s = s.replace("*", "[H]")
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        return None
    mol = Chem.RemoveHs(mol)
    return morgan_fp(mol)


# ── Functional group analysis ─────────────────────────────────────────────────

# Defined at module level because RDKit SMARTS compilation is expensive and
# fg_vector is called once per molecule in tight selection loops.
_FG_SMARTS: Dict[str, str] = {
    "imidazole": "[nH]1ccnc1",
    "pyridine": "n1ccccc1",
    "pyrimidine": "n1ccncc1",
    "triazole": "n1nncc1",
    "oxazole": "o1ccnc1",
    "thiazole": "s1ccnc1",
    "indole": "c1ccc2[nH]ccc2c1",
    "benzimidazole": "c1cnc2ccccc2n1",
    "piperazine": "N1CCNCC1",
    "piperidine": "N1CCCCC1",
    "morpholine": "N1CCOCC1",
    "amide": "C(=O)N",
    "urea": "NC(=O)N",
    "sulfonamide": "S(=O)(=O)N",
    "carbamate": "OC(=O)N",
    "ester": "C(=O)OC",
    "carboxylic_acid": "C(=O)[OH]",
    "primary_amine": "[NH2]",
    "secondary_amine": "[NH1;!$(NC=O)]",
    "hydroxyl": "[OX2H]",
    "thiol": "[SH]",
    "fluorine": "[F]",
    "chlorine": "[Cl]",
    "bromine": "[Br]",
    "acrylamide": "C=CC(=O)N",
    "chloroacetamide": "ClCC(=O)N",
    "epoxide": "C1OC1",
    "vinyl_sulfone": "C=CS(=O)(=O)",
    "peg_ether": "COCCO",
    "alkyl_chain_c4": "CCCC",
    "glutarimide": "O=C1CCC(=O)N1",
    "vhl_hydroxyproline": "[C@@H]1(O)C[C@H]",
}

_FG_QUERIES: Dict[str, Any] = {
    name: compiled
    for name, sma in _FG_SMARTS.items()
    if (compiled := Chem.MolFromSmarts(sma)) is not None
}


def fg_vector(mol: Any) -> np.ndarray:
    """Build a binary presence vector over the project's functional group library.

    Args:
        mol: RDKit Mol to profile.

    Returns:
        Float32 array of length ``len(_FG_QUERIES)``; 1 where the group is
        present in *mol*, 0 otherwise.
    """
    return np.array(
        [1 if mol.HasSubstructMatch(q) else 0 for q in _FG_QUERIES.values()],
        dtype=np.float32,
    )


def jaccard_fg(v1: np.ndarray, v2: np.ndarray) -> float:
    """Jaccard distance between two functional group presence vectors.

    Args:
        v1: FG binary vector for the first molecule.
        v2: FG binary vector for the second molecule.

    Returns:
        Distance in [0, 1]; 0 means identical FG profiles, 1 means disjoint.
    """
    intersection = float(np.dot(v1, v2))
    union = float(np.sum((v1 + v2) > 0))
    return 1.0 - (intersection / union if union > 0 else 0.0)


# ── ID generation ─────────────────────────────────────────────────────────────


def make_sequential_ids(series: pd.Series, prefix: str) -> pd.Series:
    """Assign sequential IDs (e.g. ``WH_001``) in order of first appearance.

    Order-dependent and human-readable; use ``make_hash_ids`` when run-to-run
    stability is required.

    Args:
        series: Series of SMILES strings; NaN rows receive None.
        prefix: Short label prepended to each ID, e.g. ``"WH"``, ``"LK"``, ``"E3"``.

    Returns:
        Series of ID strings aligned with *series*.
    """
    unique = series.dropna().unique()
    mapping = {smi: f"{prefix}_{i + 1:03d}" for i, smi in enumerate(unique)}
    return series.map(lambda s: mapping[s] if pd.notna(s) else None)


def make_hash_ids(series: pd.Series, prefix: str) -> pd.Series:
    """Assign hash-based IDs (e.g. ``WH_a1b2c3d4``) stable across re-runs.

    Unlike ``make_sequential_ids``, these IDs are independent of row order and
    can be reproduced from SMILES alone.

    Args:
        series: Series of SMILES strings; NaN rows receive None.
        prefix: Short label prepended to each ID, e.g. ``"WH"``, ``"LK"``, ``"E3"``.

    Returns:
        Series of ID strings aligned with *series*.
    """
    return series.map(lambda s: f"{prefix}_{smiles_hash(s)}" if pd.notna(s) else None)


# ── Surrogate-model feature computation ────────────────────────────────────────

# Shared across compute_descriptors' default calculator/n_desc args below.
DESCRIPTOR_CALCULATOR = MoleculeDescriptors.MolecularDescriptorCalculator(
    DESCRIPTOR_NAMES
)


def compute_fingerprints(
    mols: Any | List[Any], fp_size: int = 512, fp_radius: int = 3
) -> np.ndarray:
    """Compute Morgan fingerprint(s) for pre-standardized Mol(s).

    Accepts either a single Mol (or None) -- e.g. for per-row use with
    ``papply``, which calls this function element-by-element -- or an
    iterable of Mols (e.g. from ``standardize_all``).

    Args:
        mols: A single RDKit Mol (or None), or an iterable of them; None
            entries in an iterable are allowed.
        fp_size: Folded bit-vector length.
        fp_radius: Bond-hop radius from each heavy atom center.

    Returns:
        Float32 array of shape [fp_size] for a single-Mol input, or
        [len(mols), fp_size] for an iterable input; zero row(s) for None mols.
    """
    gen = rdFingerprintGenerator.GetMorganGenerator(
        radius=fp_radius, fpSize=fp_size, includeChirality=True
    )

    def _fingerprint(mol: Any) -> np.ndarray:
        if mol is None:
            return np.zeros(fp_size, dtype=np.float32)
        return gen.GetFingerprintAsNumPy(Chem.AddHs(mol)).astype(np.float32)

    if mols is None or isinstance(mols, Chem.Mol):
        return _fingerprint(mols)

    return np.vstack(
        [_fingerprint(mol) for mol in tqdm(mols, desc="Computing fingerprints", unit="mol")]
    )


def compute_descriptors(
    mols: Any | List[Any],
    calculator: Any = DESCRIPTOR_CALCULATOR,
    n_desc: int = len(DESCRIPTOR_NAMES),
) -> np.ndarray:
    """Compute RDKit descriptors for pre-standardized Mol(s).

    Accepts either a single Mol (or None) -- e.g. for per-row use with
    ``papply``, which calls this function element-by-element -- or an
    iterable of Mols (e.g. from ``standardize_all``).

    Args:
        mols: A single RDKit Mol (or None), or an iterable of them; None
            entries in an iterable are allowed.
        calculator: RDKit descriptor calculator. Defaults to the full RDKit set.
        n_desc: Number of descriptors *calculator* produces (row width for NaN rows).

    Returns:
        Float32 array of shape [n_desc] for a single-Mol input, or
        [len(mols), n_desc] for an iterable input; NaN row(s) for None mols.
    """

    def _descriptors(mol: Any) -> np.ndarray:
        if mol is None:
            return np.full(n_desc, np.nan, dtype=np.float32)
        vals = np.clip(
            np.array(calculator.CalcDescriptors(mol), dtype=np.float64), -1e4, 1e4
        )
        return vals.astype(np.float32)

    if mols is None or isinstance(mols, Chem.Mol):
        return _descriptors(mols)

    return np.vstack(
        [_descriptors(mol) for mol in tqdm(mols, desc="Computing descriptors", unit="mol")]
    )


def sanitize_matrix(X: np.ndarray) -> np.ndarray:
    """Replace inf with NaN and clip to a float32-safe range.

    Args:
        X: Feature matrix, possibly containing inf/out-of-range values.

    Returns:
        Float32 array, same shape as *X*.
    """
    X = np.where(np.isinf(X), np.nan, X)
    return np.clip(X, -1e4, 1e4).astype(np.float32)


def make_preprocessor(
    use_fingerprints: bool,
    use_descriptors: bool,
    n_fp_cols: int,
    n_desc_cols: int,
    svd_components: int = 64,
    random_state: int = 42,
) -> SkPipeline:
    """Build an unfitted sklearn preprocessing pipeline for fp/descriptor features.

    Three cases: fingerprints-only -> SVD (or passthrough); descriptors-only ->
    impute/var-threshold/scale; both -> a ColumnTransformer splitting the two
    column blocks.

    Args:
        use_fingerprints: Whether the input includes a fingerprint block.
        use_descriptors: Whether the input includes a descriptor block.
        n_fp_cols: Width of the fingerprint block (ignored if unused).
        n_desc_cols: Width of the descriptor block (ignored if unused).
        svd_components: TruncatedSVD output dims for the fingerprint block;
            0 disables SVD (passthrough).
        random_state: Seed for TruncatedSVD.

    Returns:
        Unfitted sklearn Pipeline.
    """
    # fingerprint block: optional dimensionality reduction
    fp_processor = (
        TruncatedSVD(n_components=svd_components, random_state=random_state)
        if svd_components
        else "passthrough"
    )

    # descriptor block: fill NaNs -> drop constant cols -> standardize
    desc_pipeline = SkPipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("var_thresh", VarianceThreshold(threshold=0.0)),
            ("scale", StandardScaler()),
        ]
    )

    # both active -> split columns and apply each processor to its block
    if use_fingerprints and use_descriptors:
        return SkPipeline(
            [
                (
                    "ct",
                    ColumnTransformer(
                        [
                            ("fp", fp_processor, slice(0, n_fp_cols)),
                            (
                                "desc",
                                desc_pipeline,
                                slice(n_fp_cols, n_fp_cols + n_desc_cols),
                            ),
                        ]
                    ),
                ),
            ]
        )

    # fingerprints only
    if use_fingerprints:
        return SkPipeline([("fp", fp_processor)])

    # descriptors only
    return SkPipeline([("desc", desc_pipeline)])


def get_scaffold(smiles: str, generic: bool = False) -> str:
    """Compute the Murcko scaffold for a SMILES string.

    Args:
        smiles: Input SMILES string.
        generic: If True, return the generic (element-agnostic) scaffold.

    Returns:
        Scaffold SMILES, or the original SMILES if parsing fails or the
        scaffold is empty (e.g. acyclic molecules).
    """
    mol = smiles_to_mol(smiles)
    if mol is None:  # unparseable -> fall back, don't crash
        return smiles
    if generic:
        scaffold = Chem.MolToSmiles(MakeScaffoldGeneric(mol))
    else:
        scaffold = Chem.MolToSmiles(GetScaffoldForMol(mol))
    return scaffold if len(scaffold) > 0 else smiles


def scaffold_train_test_split(
    smiles_list: List[str],
    test_size: float = 0.2,
    random_state: int = 42,
    generic: bool = False,
) -> np.ndarray:
    """Assign each molecule to 'train' or 'test' by scaffold groups.

    Molecules sharing a scaffold land in the same split (no leakage).

    Args:
        smiles_list: SMILES strings to split.
        test_size: Target fraction of molecules assigned to 'test'.
        random_state: Seed for the scaffold-group shuffle.
        generic: If True, group by the generic (element-agnostic) scaffold.

    Returns:
        Array of "train"/"test" labels aligned with *smiles_list*.
    """
    scaffold_to_idx = defaultdict(list)
    for i, smi in enumerate(smiles_list):
        scaffold_to_idx[get_scaffold(smi, generic=generic)].append(i)

    rng = np.random.default_rng(random_state)
    groups = list(scaffold_to_idx.keys())
    rng.shuffle(groups)

    n_test, test_idx = int(np.floor(test_size * len(smiles_list))), set()
    for sc in groups:
        if len(test_idx) >= n_test:
            break
        test_idx.update(scaffold_to_idx[sc])

    return np.array(
        ["test" if i in test_idx else "train" for i in range(len(smiles_list))]
    )


# ── pandas / tqdm utility ─────────────────────────────────────────────────────


def papply(series: pd.Series, func: Any, desc: str, n_jobs: int = 1) -> pd.Series:
    """Apply *func* to *series* with a labelled tqdm progress bar.

    tqdm.pandas() registers ``progress_apply`` on pd.Series but forwards all
    keyword arguments down to the underlying ``apply`` call — so ``desc`` cannot
    be passed to ``progress_apply()`` directly without it being forwarded to the
    function.  Calling ``tqdm.pandas(desc=desc)`` before each invocation is the
    safe workaround.

    Args:
        series: pandas Series to transform.
        func: Callable applied element-wise. Must be a module-level function
            (picklable by reference) when n_jobs != 1, since it is shipped to
            worker processes -- a lambda or closure will fail to pickle.
        desc: Progress bar label shown in the terminal.
        n_jobs: 1 (default) applies sequentially in-process, unchanged from
            before this parameter existed. Any other value runs func over
            series in a ProcessPoolExecutor instead -- worth it for
            RDKit-heavy functions (std_smiles, canon_smiles, ...) over
            datasets of ~1e5+ rows, where per-call C++ overhead dominates and
            multiprocessing gives a near-linear speedup; not worth the
            process-startup/IPC cost for small series. -1 uses one worker per
            CPU (os.cpu_count()); any positive value is used as-is.

    Returns:
        Transformed Series with the same index as *series*.
    """
    if n_jobs == 1:
        tqdm.pandas(desc=desc)
        return series.progress_apply(func)

    import os
    from concurrent.futures import ProcessPoolExecutor

    workers = os.cpu_count() or 1 if n_jobs < 0 else n_jobs
    chunksize = max(1, len(series) // (workers * 4))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        results = list(
            tqdm(ex.map(func, series, chunksize=chunksize), total=len(series), desc=desc)
        )
    return pd.Series(results, index=series.index)
