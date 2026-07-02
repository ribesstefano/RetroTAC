"""
csv_to_sqlite.py
================
Reads a vendor status CSV and populates a SQLite stock database with two tables:
  - vendor_stock : molecules that have a known vendor (pubchem_vendor_tag or vendor_status == 1)
  - solved_stock : molecules solved by AiZynthFinder (aizynth_solved_tag == 1)

The output tables are compatible with the SQLiteStock class used by
AiZynthFinder, which expects each table to have 'smiles' and 'inchi_key'
columns.

Input CSV format
----------------
The script auto-detects the relevant columns from the following candidates:

  Vendor column (one of):
    - vendor_status       : 1 if the molecule has a vendor
    - pubchem_vendor_tag  : 1 if the molecule was found in PubChem with a vendor

  Solved column (optional, one of):
    - aizynth_solved_tag  : 1 if the molecule was solved by AiZynthFinder

  SMILES column (one of, in priority order):
    - cap_smiles_canon
    - cap_smiles
    - component_smile_with_dummy
    - smiles

  If no SMILES column is found, the script aborts.
  If no vendor or solved column is found, the corresponding table will be empty.

Output tables
-------------
  vendor_stock : molecules with a known vendor
  solved_stock : molecules solved by AiZynthFinder (if column present)

Each table has the schema:
  id        INTEGER PRIMARY KEY AUTOINCREMENT
  smiles    TEXT NOT NULL
  inchi_key TEXT NOT NULL

Duplicates are automatically skipped at insert time via INSERT OR IGNORE.
This means the script is safe to run multiple times on the same database.

Usage
-----
    python csv_to_sqlite.py <csv_file> [db_path]

Arguments:
    csv_file  Path to the input CSV file (required).
    db_path   Path to the output SQLite database (default: stocks/aizynthfinder_stock.db).

Examples:
    python csv_to_sqlite.py vendor_results.csv
    python csv_to_sqlite.py vendor_results.csv stocks/aizynthfinder_stock.db
"""

import sys
import sqlite3
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
import argparse
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]

RDLogger.DisableLog('rdApp.*')


def create_table(cursor, table_name):
    """
    Create a stock table and its InChIKey index if they do not already exist.

    Parameters
    ----------
    cursor : sqlite3.Cursor
    table_name : str
        Name of the table to create.
    """
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS `{table_name}` (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            smiles    TEXT NOT NULL,
            inchi_key TEXT NOT NULL
        )
    """)
    cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_inchi ON `{table_name}`(inchi_key)")


def smiles_to_row(smiles_raw):
    """
    Validate a SMILES string and compute its InChIKey.

    Parameters
    ----------
    smiles_raw : str or any
        Raw SMILES value from the CSV cell.

    Returns
    -------
    tuple(str, str) or None
        (canonical_smiles, inchi_key) if valid, None otherwise.
    """
    try:
        smiles = str(smiles_raw).strip()
        if not smiles or smiles.lower() == 'nan':
            return None
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None and mol.GetNumAtoms() > 0:
            inchi_key = Chem.MolToInchiKey(mol)
            if inchi_key:
                return (smiles, inchi_key)
    except Exception:
        pass
    return None


def csv_to_sqlite(csv_file, db_path, chunk_size=10000):
    """
    Read a vendor status CSV and populate vendor_stock and solved_stock tables.

    Auto-detects available columns for vendor status, solved status, and SMILES.
    Molecules with a vendor are written to vendor_stock; molecules solved by
    AiZynthFinder (if the column exists) are written to solved_stock.

    Parameters
    ----------
    csv_file : str
        Path to the input CSV file.
    db_path : str
        Path to the output SQLite database. Created if it does not exist.
    chunk_size : int, optional
        Number of CSV rows to process per iteration (default: 10000).
    """
    print(f"Input : {csv_file}", flush=True)
    print(f"DB    : {db_path}  ->  tables: vendor_stock, solved_stock", flush=True)
    print(flush=True)

    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    conn   = sqlite3.connect(db_path)
    cursor = conn.cursor()

    create_table(cursor, "vendor_stock")
    create_table(cursor, "solved_stock")
    conn.commit()

    chunk_reader = pd.read_csv(csv_file, chunksize=chunk_size, low_memory=False)

    total_vendor  = 0
    total_solved  = 0
    total_skipped = 0

# Detect columns from the first chunk and process all chunks
    first_chunk    = None
    vendor_col     = solved_col = smiles_col = None
    total_vendor   = 0
    total_solved   = 0
    total_skipped  = 0

    for chunk in pd.read_csv(csv_file, chunksize=chunk_size, low_memory=False):

        if first_chunk is None:
            first_chunk = chunk
            cols        = set(chunk.columns)

            vendor_col = next((c for c in ['vendor_status', 'pubchem_vendor_tag'] if c in cols), None)
            solved_col = next((c for c in ['aizynth_solved_tag'] if c in cols), None)
            smiles_col = next((c for c in ['cap_smiles_canon', 'cap_smiles',
                                           'component_smile_with_dummy', 'smiles'] if c in cols), None)

            print(f"  Detected columns → vendor: {vendor_col}  |  solved: {solved_col}  |  smiles: {smiles_col}", flush=True)

            if smiles_col is None:
                print("✗ No SMILES column found, aborting.")
                conn.close()
                sys.exit(1)
            if vendor_col is None:
                print("⚠ No vendor column found, vendor_stock will be empty.")
            if solved_col is None:
                print("⚠ No solved column found, solved_stock will be empty.")

        vendor_rows = []
        solved_rows = []
        skipped     = 0

        for _, row in chunk.iterrows():
            has_vendor = str(row.get(vendor_col, '')).strip() in ('1', '1.0') if vendor_col else False
            is_solved  = str(row.get(solved_col, '')).strip() in ('1', '1.0') if solved_col else False
            smiles_raw = row.get(smiles_col)

            mol_row = smiles_to_row(smiles_raw)
            if mol_row is None:
                skipped += 1
                continue

            if has_vendor:
                vendor_rows.append(mol_row)
            elif is_solved:
                solved_rows.append(mol_row)
            else:
                skipped += 1

        if vendor_rows:
            cursor.executemany("INSERT OR IGNORE INTO `vendor_stock` (smiles, inchi_key) VALUES (?, ?)", vendor_rows)
        if solved_rows:
            cursor.executemany("INSERT OR IGNORE INTO `solved_stock` (smiles, inchi_key) VALUES (?, ?)", solved_rows)
        conn.commit()

        total_vendor  += len(vendor_rows)
        total_solved  += len(solved_rows)
        total_skipped += skipped

        print(f"✓  vendor={len(vendor_rows)}  solved={len(solved_rows)}  skipped={skipped}", flush=True)

    v_count = cursor.execute("SELECT COUNT(*) FROM vendor_stock").fetchone()[0]
    s_count = cursor.execute("SELECT COUNT(*) FROM solved_stock").fetchone()[0]

    print(f"\n{'='*55}")
    print(f"  vendor_stock : {v_count:>10,} molecules")
    print(f"  solved_stock : {s_count:>10,} molecules")
    print(f"  Skipped      : {total_skipped:>10,}")
    print(f"{'='*55}")
    print(f"\n  DB saved to: {db_path}")

    conn.close()


def main():
    parser = argparse.ArgumentParser(description="Load vendor status CSV into a SQLite stock database")
    parser.add_argument("input", type=str,
                        help="Path to input CSV file")
    parser.add_argument("--db_path", type=str, default=str(_PROJECT_ROOT / "data/external/aizynthfinder_stock.db"),
                        help="Path to the output SQLite database.")
    args = parser.parse_args()
    csv_to_sqlite(args.input, args.db_path)


if __name__ == "__main__":
    main()