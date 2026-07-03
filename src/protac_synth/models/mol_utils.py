"""
mol_utils.py
============
Framework-agnostic molecule handling shared by all models.
"""
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles, MakeScaffoldGeneric
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

def compute_fingerprints(smiles_list, fp_size: int = 512, fp_radius: int = 3) -> np.ndarray:
    """Morgan fingerprints for a list of SMILES; zero-vector row for unparseable mols.
    Returns array shape [len(smiles_list), fp_size].
    """
    gen = AllChem.GetMorganGenerator(radius=fp_radius, fpSize=fp_size, includeChirality=True)
    fps = []
    for smi in smiles_list:
        mol = standardize(smi)
        if mol is None:
            fps.append(np.zeros(fp_size, dtype=np.float32))
        else:
            mol = Chem.AddHs(mol)
            fp = gen.GetFingerprintAsNumPy(mol).astype(np.float32)
            fps.append(fp)
    return np.vstack(fps)

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
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    if generic:
        scaffold = Chem.MolToSmiles(MakeScaffoldGeneric(mol))
    else:
        from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol
        scaffold = Chem.MolToSmiles(GetScaffoldForMol(mol))
    return scaffold if len(scaffold) > 0 else smiles

def compute_descriptors(smiles_list, calculator=DESCRIPTOR_CALCULATOR,
                        n_desc: int = len(DESCRIPTOR_NAMES)) -> np.ndarray:
    """RDKit descriptors for a list of SMILES; NaN row for unparseable mols.
    Returns array shape [len(smiles_list), n_desc], float32."""
    rows = []
    for smi in smiles_list:
        mol = standardize(smi)
        if mol is None:
            rows.append(np.full(n_desc, np.nan, dtype=np.float32))
        else:
            vals = np.array(calculator.CalcDescriptors(mol), dtype=np.float64)
            vals = np.clip(vals, -1e4, 1e4)
            rows.append(vals.astype(np.float32))
    return np.vstack(rows)