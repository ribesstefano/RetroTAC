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
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize
from tqdm import tqdm

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

def morgan_fp(mol: Any, radius: int = 3, nbits: int = 2048) -> Any:
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
    s = re.sub(r'\[\*:\d+\]', '[H]', s)
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
    "imidazole":          "[nH]1ccnc1",
    "pyridine":           "n1ccccc1",
    "pyrimidine":         "n1ccncc1",
    "triazole":           "n1nncc1",
    "oxazole":            "o1ccnc1",
    "thiazole":           "s1ccnc1",
    "indole":             "c1ccc2[nH]ccc2c1",
    "benzimidazole":      "c1cnc2ccccc2n1",
    "piperazine":         "N1CCNCC1",
    "piperidine":         "N1CCCCC1",
    "morpholine":         "N1CCOCC1",
    "amide":              "C(=O)N",
    "urea":               "NC(=O)N",
    "sulfonamide":        "S(=O)(=O)N",
    "carbamate":          "OC(=O)N",
    "ester":              "C(=O)OC",
    "carboxylic_acid":    "C(=O)[OH]",
    "primary_amine":      "[NH2]",
    "secondary_amine":    "[NH1;!$(NC=O)]",
    "hydroxyl":           "[OX2H]",
    "thiol":              "[SH]",
    "fluorine":           "[F]",
    "chlorine":           "[Cl]",
    "bromine":            "[Br]",
    "acrylamide":         "C=CC(=O)N",
    "chloroacetamide":    "ClCC(=O)N",
    "epoxide":            "C1OC1",
    "vinyl_sulfone":      "C=CS(=O)(=O)",
    "peg_ether":          "COCCO",
    "alkyl_chain_c4":     "CCCC",
    "glutarimide":        "O=C1CCC(=O)N1",
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


# ── pandas / tqdm utility ─────────────────────────────────────────────────────

def papply(series: pd.Series, func: Any, desc: str) -> pd.Series:
    """Apply *func* to *series* with a labelled tqdm progress bar.

    tqdm.pandas() registers ``progress_apply`` on pd.Series but forwards all
    keyword arguments down to the underlying ``apply`` call — so ``desc`` cannot
    be passed to ``progress_apply()`` directly without it being forwarded to the
    function.  Calling ``tqdm.pandas(desc=desc)`` before each invocation is the
    safe workaround.

    Args:
        series: pandas Series to transform.
        func: Callable applied element-wise.
        desc: Progress bar label shown in the terminal.

    Returns:
        Transformed Series with the same index as *series*.
    """
    tqdm.pandas(desc=desc)
    return series.progress_apply(func)
