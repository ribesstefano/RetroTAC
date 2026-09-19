"""
Stage 1, step 6 — assemble the three preprocessed CSVs into a single SQLite database.

Three tables are created (existing database is overwritten):
  protac               — one row per PROTAC molecule
  component_cap        — one row per (component, cap variant) pair
  component_cid_vendor — one row per (component, cap) PubChem result

Join paths:
  protac.{warhead,linker,e3_ligase_ligand}_id  →  component_cap.component_id
  component_cap.(component_id, cap_smiles)     ↔  component_cid_vendor.(component_id, cap_smiles)

Indexes are built on all foreign-key columns and on cap_smiles / CID to
support the most common filter and join patterns downstream.

I/O
---
    in :  data/preprocessed/protac_master.csv
          data/preprocessed/component_capped.csv
          data/preprocessed/component_cid_vendor.csv
    out:  data/preprocessed/pipeline.db
"""

import argparse
import sqlite3
from pathlib import Path

import pandas as pd
from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

CHUNKSIZE = 10_000


def _write_table(conn: sqlite3.Connection, df: pd.DataFrame, table: str) -> None:
    """Load a DataFrame into SQLite in chunks, reporting progress with tqdm.

    Args:
        conn: Open SQLite connection.
        df: DataFrame to write.
        table: Destination table name; replaced if it already exists.
    """
    chunks = range(0, len(df), CHUNKSIZE)
    for i, start in enumerate(tqdm(chunks, desc=f"  {table}", unit="chunk")):
        df.iloc[start : start + CHUNKSIZE].to_sql(
            table, conn,
            if_exists="replace" if i == 0 else "append",
            index=False,
        )


def _build_indexes(conn: sqlite3.Connection) -> None:
    """Create covering indexes for FK joins and common filter columns.

    Args:
        conn: Open SQLite connection; indexes are committed before returning.
    """
    stmts = [
        "CREATE INDEX IF NOT EXISTS idx_protac_warhead_id    ON protac(warhead_id)",
        "CREATE INDEX IF NOT EXISTS idx_protac_linker_id     ON protac(linker_id)",
        "CREATE INDEX IF NOT EXISTS idx_protac_e3_id         ON protac(e3_ligase_ligand_id)",
        "CREATE INDEX IF NOT EXISTS idx_cap_component_id     ON component_cap(component_id)",
        "CREATE INDEX IF NOT EXISTS idx_cap_cap_smiles       ON component_cap(cap_smiles)",
        "CREATE INDEX IF NOT EXISTS idx_vendor_component_id  ON component_cid_vendor(component_id)",
        "CREATE INDEX IF NOT EXISTS idx_vendor_cap_smiles    ON component_cid_vendor(cap_smiles)",
        "CREATE INDEX IF NOT EXISTS idx_vendor_cid           ON component_cid_vendor(CID)",
    ]
    for stmt in tqdm(stmts, desc="  indexes"):
        conn.execute(stmt)
    conn.commit()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Populated ``argparse.Namespace``.
    """
    p = argparse.ArgumentParser(
        description="Merge preprocessed CSVs into a single SQLite database.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--protac", type=Path, default=_PROJECT_ROOT / "data/preprocessed/protac_master.csv",
                   help="PROTAC master table CSV.")
    p.add_argument("--capped", type=Path, default=_PROJECT_ROOT / "data/preprocessed/component_capped.csv",
                   help="Capped component variants CSV.")
    p.add_argument("--vendor", type=Path, default=_PROJECT_ROOT / "data/preprocessed/component_cid_vendor.csv",
                   help="PubChem CID and vendor status CSV.")
    p.add_argument("--output", type=Path, default=_PROJECT_ROOT / "data/preprocessed/pipeline.db",
                   help="SQLite output path (overwritten if it exists).")
    return p.parse_args()


def main(
    protac_path: Path,
    capped_path: Path,
    vendor_path: Path,
    db_path: Path,
) -> None:
    """Read the three preprocessed CSVs and assemble them into one SQLite database.

    Args:
        protac_path: PROTAC master table CSV path.
        capped_path: Capped component variants CSV path.
        vendor_path: PubChem CID and vendor status CSV path.
        db_path: Destination SQLite path; overwritten if it already exists.
    """
    print("Reading CSVs...")
    protac = pd.read_csv(protac_path)
    capped = pd.read_csv(capped_path)
    vendor = pd.read_csv(vendor_path)

    # Normalise nullable integer columns so SQLite stores them as INTEGER / NULL
    for col in ("CID", "vendor_status"):
        vendor[col] = vendor[col].astype(pd.Int64Dtype())

    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(db_path)
    try:
        print("Writing tables...")
        _write_table(conn, protac, "protac")
        _write_table(conn, capped, "component_cap")
        _write_table(conn, vendor, "component_cid_vendor")

        print("Building indexes...")
        _build_indexes(conn)
    finally:
        conn.close()

    size_mb = db_path.stat().st_size / 1024 ** 2
    print(f"\nSaved → {db_path}  ({size_mb:.1f} MB)")
    print(f"  protac               : {len(protac):>7,} rows")
    print(f"  component_cap        : {len(capped):>7,} rows")
    print(f"  component_cid_vendor : {len(vendor):>7,} rows")


if __name__ == "__main__":
    args = parse_args()
    main(args.protac, args.capped, args.vendor, args.output)
