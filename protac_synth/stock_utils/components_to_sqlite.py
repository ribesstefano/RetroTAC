"""
components_to_sqlite.py
=======================
Reads a components CSV and populates a SQLite stock database with one table
per component type (warhead, e3_ligase, linker), filtered to only include
rows where final_solved_tag == 1, 2 or 3.

The output tables are compatible with the SQLiteStock class used by
AiZynthFinder, which expects each table to have 'smiles' and 'inchi_key'
columns.

Input CSV format
----------------
Expected columns (others are ignored):
  - components        : component type label ('warhead', 'e3ligase', 'linker')
  - cap_smiles_canon  : canonical SMILES of the capped component (used as stock SMILES)
  - final_solved_tag  : 1 if the component should be included, 0 otherwise

Output tables
-------------
  warhead    : capped warhead SMILES with final_solved_tag == 1, 2 or 3
  e3_ligase  : capped E3 ligase SMILES with final_solved_tag == 1, 2 or 3
  linker     : capped linker SMILES with final_solved_tag == 1, 2 or 3

Each table has the schema:
  id        INTEGER PRIMARY KEY AUTOINCREMENT
  smiles    TEXT NOT NULL
  inchi_key TEXT NOT NULL UNIQUE

Duplicates are automatically skipped at insert time via INSERT OR IGNORE on
the UNIQUE inchi_key constraint. This means the script is safe to run multiple
times on the same database — already-present molecules will not be duplicated.

Usage
-----
    python components_to_sqlite.py <csv_file> [db_path]

Examples:
    python components_to_sqlite.py components.csv stocks/aizynthfinder_stock.db
    python components_to_sqlite.py components.csv   # uses default DB path

Arguments:
    csv_file  Path to the input components CSV (required).
    db_path   Path to the output SQLite database (default: stocks/aizynthfinder_stock.db).
"""

import argparse
import sqlite3
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]

RDLogger.DisableLog('rdApp.*')

# Maps the value found in the 'components' column to the SQLite table name.
# Extend this dict if your CSV uses different labels.
COMPONENT_TABLE_MAP = {
    "warhead":  "warhead",
    "e3ligase": "e3_ligase",
    "e3_ligase": "e3_ligase",   
    "linker":   "linker",
}

SMILES_COL      = "cap_smiles_canon"
COMPONENT_COL   = "components"
SOLVED_TAG_COL  = "final_solved_tag"


def create_table(cursor, table_name):
    """
    Create a stock table and its InChIKey index if they do not already exist.

    Parameters
    ----------
    cursor : sqlite3.Cursor
    table_name : str
        Name of the table to create. Must match the name used when loading
        the stock in AiZynthFinder (e.g. 'warhead', 'e3_ligase', 'linker').
    """
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS `{table_name}` (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            smiles    TEXT NOT NULL,
            inchi_key TEXT NOT NULL UNIQUE
        )
    """)
    cursor.execute(
        f"CREATE INDEX IF NOT EXISTS idx_{table_name}_inchi ON `{table_name}`(inchi_key)"
    )


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


def components_to_sqlite(csv_file, db_path, chunk_size=10000):
    """
    Read the components CSV and write solved components into per-type stock tables.

    Only rows where final_solved_tag == 1 are written. Each component type
    (warhead, e3_ligase, linker) gets its own table. Rows with unrecognised
    component labels or invalid SMILES are skipped and counted separately.

    Parameters
    ----------
    csv_file : str
        Path to the input components CSV file.
    db_path : str
        Path to the output SQLite database. Created if it does not exist.
    chunk_size : int, optional
        Number of CSV rows to process per iteration (default: 10000).
    """
    print(f"Input : {csv_file}", flush=True)
    print(f"DB    : {db_path}", flush=True)
    print(f"Tables: {', '.join(sorted(set(COMPONENT_TABLE_MAP.values())))}", flush=True)
    print(flush=True)

    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    conn   = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create one table per unique target table name
    for table_name in sorted(set(COMPONENT_TABLE_MAP.values())):
        create_table(cursor, table_name)
    conn.commit()

    chunk_reader = pd.read_csv(csv_file, chunksize=chunk_size, low_memory=False)

    # Counters per table + global skipped
    totals  = {t: 0 for t in set(COMPONENT_TABLE_MAP.values())}
    skipped = 0

    for chunk_num, chunk in enumerate(chunk_reader):
        start = chunk_num * chunk_size
        print(f"  Processing rows [{start:,} – {start + len(chunk) - 1:,}]...",
              end=" ", flush=True)

        # Bucket rows by target table
        batch: dict[str, list] = {t: [] for t in set(COMPONENT_TABLE_MAP.values())}
        chunk_skipped = 0

        for _, row in chunk.iterrows():
            # Only include rows with final_solved_tag == 1
            solved = str(row.get(SOLVED_TAG_COL, '')).strip() in ('1', '1.0','2','2.0','3','3.0')
            if not solved:
                chunk_skipped += 1
                continue

            # Map component label to table name
            component_label = str(row.get(COMPONENT_COL, '')).strip().lower()
            table_name = COMPONENT_TABLE_MAP.get(component_label)
            if table_name is None:
                chunk_skipped += 1
                continue

            mol_row = smiles_to_row(row.get(SMILES_COL))
            if mol_row is None:
                chunk_skipped += 1
                continue

            batch[table_name].append(mol_row)

        # Write each bucket to its table
        chunk_counts = {}
        for table_name, rows in batch.items():
            if rows:
                cursor.executemany(
                    f"INSERT OR IGNORE INTO `{table_name}` (smiles, inchi_key) VALUES (?, ?)",
                    rows
                )
                totals[table_name] += len(rows)
            chunk_counts[table_name] = len(rows)

        conn.commit()
        skipped += chunk_skipped

        counts_str = "  ".join(f"{t}={n}" for t, n in chunk_counts.items())
        print(f"✓  {counts_str}  skipped={chunk_skipped}", flush=True)

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    for table_name in sorted(totals):
        db_count = cursor.execute(f"SELECT COUNT(*) FROM `{table_name}`").fetchone()[0]
        print(f"  {table_name:<12}: {db_count:>10,} molecules")
    print(f"  {'Skipped':<12}: {skipped:>10,}")
    print(f"{'='*55}")
    print(f"\n  DB saved to: {db_path}")
    print(f"\nTo load in AiZynthFinder:")
    for table_name in sorted(totals):
        print(f"  SQLiteStock('{db_path}', '{table_name}')")

    conn.close()

def main():
    parser = argparse.ArgumentParser(description="Upload components (warhead, e3 ligase and linker) to stocks database")
    parser.add_argument("input", type=str,
                        help="Path to CSV input file")
    parser.add_argument("--db_path", type=str, default=str(_PROJECT_ROOT / "data/external/aizynthfinder_stock.db"),
                        help="Path to the output SQLite database.")
    args = parser.parse_args()
    components_to_sqlite(args.input, args.db_path)

if __name__ == "__main__":
    main()

    