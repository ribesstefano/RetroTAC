"""
Stage 1, step 1 — decompose TPDDB PROTACs into components via PROTAC-Splitter.

Input:
    tpddb_protacs(2).csv         — full TPDDB dataset (SMILES column used)
    dataset-curated-held-out.csv — held-out set; overlap removed before splitting

Output:
    tpddb_split_raw_complete.csv — raw PROTAC-Splitter output
    columns: smiles, __index_level_0__, default_pred_n0, model_name
    (fragment extraction into warhead/linker/E3 is done in step 2)

Note: Run only once — PROTAC-Splitter is slow.
"""

from pathlib import Path

import pandas as pd
from protac_splitter import split_protac
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")

# ── Paths — edit BASE to local path ─────────────────────
BASE = Path(__file__).parent.parent / "data"
# BASE = "/mimer/NOBACKUP/groups/naiss2023-6-290/tingtingmo/data"

TPDDB_FILE = BASE / "tpddb_protacs(2).csv"
HELD_FILE = BASE / "dataset-curated-held-out.csv"
OUTPUT_FILE = BASE / "tpddb_split_raw_complete.csv"

# ── Load ──────────────────────────────────────────────────────────────────────
tpddb = (
    pd.read_csv(TPDDB_FILE, usecols=["SMILES", "TPD_ID", "Database"])
    .rename(columns={"SMILES": "smiles", "TPD_ID": "tpd_id", "Database": "database"})
)

held_smiles = pd.read_csv(HELD_FILE, usecols=["PROTAC SMILES"])["PROTAC SMILES"]

# ── Remove overlap with held-out before splitting ─────────────────────────────
overlap = set(tpddb["smiles"]) & set(held_smiles)
print(f"Overlapping molecules removed from TPDDB: {len(overlap)}")

tpddb_clean = tpddb[~tpddb["smiles"].isin(overlap)].copy()
print(f"Original TPDDB size : {len(tpddb)}")
print(f"Cleaned  TPDDB size : {len(tpddb_clean)}")

# ── Run PROTAC-Splitter ───────────────────────────────────────────────────────
print(f"\nRunning split_protac on {len(tpddb_clean)} molecules (this may take a while)…")
split_df = split_protac(tpddb_clean, protac_smiles_col="smiles")

if "default_pred_n0" not in split_df.columns:
    raise ValueError("'default_pred_n0' not found in split_protac output — check splitter version.")

# ── Save ──────────────────────────────────────────────────────────────────────
split_df.to_csv(OUTPUT_FILE, index=False)
print(f"\nSaved → {OUTPUT_FILE}  ({len(split_df)} rows)")
print(f"Columns: {split_df.columns.tolist()}")
