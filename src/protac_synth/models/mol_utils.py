"""
mol_utils.py
============
Framework-agnostic molecule handling shared by all models.
"""
import numpy as np
from collections import defaultdict
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles, MakeScaffoldGeneric, GetScaffoldForMol
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD

RDLogger.DisableLog('rdApp.*')

DESCRIPTOR_NAMES      = [name for name, _ in Descriptors._descList if name != "Ipc"]
DESCRIPTOR_CALCULATOR = MoleculeDescriptors.MolecularDescriptorCalculator(DESCRIPTOR_NAMES)

def standardize(smiles: str, uncharge: bool = False):
    """Parse + standardize a SMILES into an RDKit Mol (largest fragment, optional uncharge).
    Returns Chem.Mol or None on failure.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    try:
        lfg = rdMolStandardize.LargestFragmentChooser()
        mol = lfg.choose(mol)
        if uncharge:
            u = rdMolStandardize.Uncharger()
            mol = u.uncharge(mol)
    except Exception:
        return None
    return mol

def standardize_all(smiles_list, uncharge: bool = False):
    """Standardize a list of SMILES once -> list of RDKit Mol (or None)."""
    return [standardize(s, uncharge) for s in smiles_list]

def compute_fingerprints(mols, fp_size: int = 512, fp_radius: int = 3) -> np.ndarray:
    """Morgan fingerprints for pre-standardized mols; zero-vector row for None."""
    gen = AllChem.GetMorganGenerator(radius=fp_radius, fpSize=fp_size, includeChirality=True)
    fps = []
    for mol in tqdm(mols, desc="Computing fingerprints", unit="mol"):
        if mol is None:
            fps.append(np.zeros(fp_size, dtype=np.float32))
        else:
            fps.append(gen.GetFingerprintAsNumPy(Chem.AddHs(mol)).astype(np.float32))
    return np.vstack(fps)


def compute_descriptors(mols, calculator=DESCRIPTOR_CALCULATOR,
                        n_desc: int = len(DESCRIPTOR_NAMES)) -> np.ndarray:
    """RDKit descriptors for pre-standardized mols; NaN row for None."""
    rows = []
    for mol in tqdm(mols, desc="Computing descriptors", unit="mol"):
        if mol is None:
            rows.append(np.full(n_desc, np.nan, dtype=np.float32))
        else:
            vals = np.clip(np.array(calculator.CalcDescriptors(mol), dtype=np.float64), -1e4, 1e4)
            rows.append(vals.astype(np.float32))
    return np.vstack(rows)


def sanitize_matrix(X: np.ndarray) -> np.ndarray:
    """Replace inf with NaN, clip to float32-safe range."""
    X = np.where(np.isinf(X), np.nan, X)
    return np.clip(X, -1e4, 1e4).astype(np.float32)


def make_preprocessor(use_fingerprints: bool, use_descriptors: bool,
                      n_fp_cols: int, n_desc_cols: int,
                      svd_components: int = 64, random_state: int = 42) -> SkPipeline:
    """Unfitted preprocessing pipeline. Three cases:
       FP-only -> SVD (or passthrough); desc-only -> impute/varthresh/scale;
       both -> ColumnTransformer splitting the two column blocks.
    """
    # fingerprint block: optional dimensionality reduction
    fp_processor = (
        TruncatedSVD(n_components=svd_components, random_state=random_state)
        if svd_components else 'passthrough'
    )

    # descriptor block: fill NaNs -> drop constant cols -> standardize
    desc_pipeline = SkPipeline([
        ('impute',     SimpleImputer(strategy='median')),
        ('var_thresh', VarianceThreshold(threshold=0.0)),
        ('scale',      StandardScaler()),
    ])

    # both active -> split columns and apply each processor to its block
    if use_fingerprints and use_descriptors:
        return SkPipeline([
            ('ct', ColumnTransformer([
                ('fp',   fp_processor,  slice(0, n_fp_cols)),
                ('desc', desc_pipeline, slice(n_fp_cols, n_fp_cols + n_desc_cols)),
            ])),
        ])

    # fingerprints only
    if use_fingerprints:
        return SkPipeline([('fp', fp_processor)])

    # descriptors only
    return SkPipeline([('desc', desc_pipeline)])

def get_scaffold(smiles: str, generic: bool = False) -> str:
    """Compute the Murcko scaffold for a SMILES string.

    Returns the scaffold SMILES, or the original SMILES if parsing fails
    or the scaffold is empty (e.g. acyclic molecules).
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:                      # unparseable -> fall back, don't crash
        return smiles
    if generic:
        scaffold = Chem.MolToSmiles(MakeScaffoldGeneric(mol))
    else:
        scaffold = Chem.MolToSmiles(GetScaffoldForMol(mol))
    return scaffold if len(scaffold) > 0 else smiles

def scaffold_train_test_split(smiles_list, test_size=0.2, random_state=42, generic=False):
    """Assign each molecule to 'train' or 'test' by scaffold groups.
    Molecules sharing a scaffold land in the same split (no leakage)."""
    scaffold_to_idx = defaultdict(list)
    for i, smi in enumerate(smiles_list):
        scaffold_to_idx[get_scaffold(smi, generic=generic)].append(i)

    rng    = np.random.default_rng(random_state)
    groups = list(scaffold_to_idx.keys())
    rng.shuffle(groups)

    n_test, test_idx = int(np.floor(test_size * len(smiles_list))), set()
    for sc in groups:
        if len(test_idx) >= n_test:
            break
        test_idx.update(scaffold_to_idx[sc])

    return np.array(["test" if i in test_idx else "train" for i in range(len(smiles_list))])