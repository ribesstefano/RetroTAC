"""
cxsmiles_to_sqlite.py
Reads a cxsmiles file and populates a SQLite stock database with one table that
includes all of the components present in the cxsmiles file.
The output table is compatible with the SQLiteStock class used by AiZynthFinder,
which expects each table to have 'smiles' and 'inchi_key' columns.

Input cxsmiles format
---------------------
Expected columns:
  - SMILES string : canonical SMILES of the molecule. The syntax is slightly different than traditional SMILES encoding
  - Extended features: additional molecular features

Output table
------------
  stock collection: collection with the corresponding molecules from the cxsmiles file

Each table has the schema:
  id        INTEGER PRIMARY KEY AUTOINCREMENT
  smiles    TEXT NOT NULL
  inchi_key TEXT NOT NULL UNIQUE

Duplicates are automatically skipped at insert time via INSERT OR IGNORE on
the UNIQUE inchi_key constraint. This means the script is safe to run multiple
times on the same database — already-present molecules will not be duplicated.

Usage
-----
    python convert_to_sqlite.py <cxsmiles_file> <table_name> [db_path]

Arguments:
    cxsmiles_file Path to the input components CSV (required).
    table_name    Name of the new table (required).
    db_path       Path to the output SQLite database (default: stocks/aizynthfinder_stock.db).
"""

import argparse
import sqlite3
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog('rdApp.*')
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def cxsmiles_to_sqlite(
    cxsmiles_file: str, collection_name: str, db_path: str, chunk_size: int = 10000
) -> None:
    """Convert a CXSMILES file to a SQLite stock database table.

    Args:
        cxsmiles_file (str):   Path to the input CXSMILES file (tab-separated).
        collection_name (str): Name of the table to create or append to.
        db_path (str):         Path to the output SQLite database file.
        chunk_size (int):      Number of rows to process per chunk (default: 10000).

    Returns:
        None. Results are written directly to the SQLite database.
    """
    print(f"Converting {cxsmiles_file} -> {db_path}:{collection_name}", flush=True)

    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS `{collection_name}` (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            smiles    TEXT NOT NULL,
            inchi_key TEXT NOT NULL
        )
    """)
    cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{collection_name}_inchi ON `{collection_name}`(inchi_key)")
    conn.commit()

    chunk_reader = pd.read_csv(cxsmiles_file, sep='\t', chunksize=chunk_size, low_memory=False)

    total_valid   = 0
    total_skipped = 0

    for chunk_num, chunk in enumerate(chunk_reader):
        start = chunk_num * chunk_size
        print(f"  Processing [{start:,} - {start + len(chunk):,}]...", end=" ", flush=True)

        rows    = []
        skipped = 0

        for _, row in chunk.iterrows():
            try:
                smiles = str(row['smiles']).strip()
                if not smiles:
                    skipped += 1
                    continue

                mol = Chem.MolFromSmiles(smiles)
                if mol is not None and mol.GetNumAtoms() > 0:
                    inchi_key = Chem.MolToInchiKey(mol)
                    if inchi_key:
                        rows.append((smiles, inchi_key))
                        total_valid += 1
                    else:
                        skipped += 1
                else:
                    skipped += 1
            except Exception:
                skipped += 1

        if rows:
            cursor.executemany(f"INSERT OR IGNORE INTO `{collection_name}` (smiles, inchi_key) VALUES (?, ?)", rows)

        total_skipped += skipped
        print(f"✓ ({len(rows)} valid, {skipped} skipped)", flush=True)

    count = cursor.execute(f"SELECT COUNT(*) FROM {collection_name}").fetchone()[0]
    print(f"\n✓ Done: {count:,} molecules in {db_path}:{collection_name}")
    conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload molecules from cxsmiles files to stocks database")
    parser.add_argument("input", type=str,
                        help="Path to CXSMILES input file")
    parser.add_argument("table_name", type=str,
                        help="Name of the table to add the molecules")
    parser.add_argument("--db_path", type=str, default=str(_PROJECT_ROOT / "data/external/aizynthfinder_stock.db"),
                        help="Path to the output SQLite database")
    args = parser.parse_args()
    cxsmiles_to_sqlite(args.input, args.table_name, args.db_path)


if __name__ == "__main__":
    main()
