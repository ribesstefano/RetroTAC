"""
sqlite_stock.py
===============
A SQLite-backed stock for AiZynthFinder.

``SQLiteStock`` adapts one table of a SQLite database (with ``smiles`` and
``inchi_key`` columns) into the interface AiZynthFinder expects of a stock —
fast ``in``/``len``/iteration via the InChIKey index. This lets very large stock
collections (ZINC, Enamine REAL, solved components, vendor hits) be queried on
disk instead of loaded into memory, and is used by ``get_scores.py`` and
``component_routes_scores.py``.

Library module — imported, not run directly. Tables are populated by the
builders in ``protac_synthesizability/stock/``.

Example
-------
    from sqlite_stock import SQLiteStock
    finder.config.stock.load(SQLiteStock("aizynthfinder_stock.db", "zinc"), "zinc")
"""

import sqlite3
from typing import Optional

from aizynthfinder.context.stock.queries import StockQueryMixin


class SQLiteStock(StockQueryMixin):
    """Expose one table of a SQLite DB as an AiZynthFinder stock query."""


    def __init__(self, path: str, table: str):
        self._path  = path
        self._table = table
        self._conn  = sqlite3.connect(path, check_same_thread=False)
        self._count: Optional[int] = None

    def __contains__(self, mol):
        inchi_key = mol.inchi_key
        result = self._conn.execute(
            f"SELECT 1 FROM `{self._table}` WHERE inchi_key=? LIMIT 1",
            (inchi_key,)
        ).fetchone()
        return result is not None

    def __len__(self):
        # AiZynthFinder 4.4.1 calls len() on every selected stock after select()
        # just for logging. SELECT COUNT(*) is O(N) on large tables; MAX(rowid)
        # reads a single B-tree leaf and returns in microseconds on any table size.
        # We cache it because select() may be called more than once per run.
        if self._count is None:
            row = self._conn.execute(
                f"SELECT MAX(rowid) FROM `{self._table}`"
            ).fetchone()
            self._count = row[0] or 0
        return self._count

    def __iter__(self):
        for row in self._conn.execute(f"SELECT smiles FROM `{self._table}`"):
            yield row[0]