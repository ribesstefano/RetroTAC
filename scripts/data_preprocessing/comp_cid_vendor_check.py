"""
Stage 1, step 5 — commercial availability check via PubChem.

For each unique capped-component SMILES, queries the PubChem PUG REST API in
two passes: SMILES → CID, then CID → vendor status. A component is a "vendor
hit" if PubChem lists at least one commercial supplier for its CID.

Both passes write an intermediate CSV cache so the script can be safely
interrupted and resumed without re-querying already-processed entries.

I/O
---
    in :  data/processed/component_capped.csv
    out:  data/processed/component_smiles_to_cid.csv    (SMILES→CID cache)
          data/processed/component_cid_to_vendor.csv    (CID→vendor cache)
          data/processed/component_check_cid_vendor.csv (component_id, component,
              cap_smiles, CID, vendor_status)
"""

import argparse
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import pandas as pd
import requests
from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
PUBCHEM_VIEW = "https://pubchem.ncbi.nlm.nih.gov/rest/pug_view"

SMILES_COL = "cap_smiles"
CID_COL = "CID"


# ── PubChem API ───────────────────────────────────────────────────────────────


def pubchem_smiles_to_cid(smiles: str, sleep_s: float = 0.25, retries: int = 1) -> Optional[int]:
    """Look up the PubChem CID for a single SMILES string.

    Args:
        smiles: Query SMILES.
        sleep_s: Delay after each attempt (seconds).
        retries: Extra retry attempts on transient server errors (429, 5xx).

    Returns:
        The first matching CID, or None if the compound is not in PubChem or
        all attempts fail.
    """
    url = f"{PUBCHEM_BASE}/compound/smiles/cids/JSON"
    for attempt in range(retries + 1):
        try:
            r = requests.get(url, params={"smiles": smiles}, timeout=(5, 12))
            time.sleep(sleep_s)
            if r.status_code == 200:
                cids = r.json().get("IdentifierList", {}).get("CID", [])
                return cids[0] if cids else None
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(1.0 * (attempt + 1))
                continue
            return None
        except Exception:
            time.sleep(1.0 * (attempt + 1))
    return None


def _post_smiles_batch(
    smiles_batch: List[str],
    sleep_s: float,
    retries: int,
) -> Optional[List[Optional[int]]]:
    """POST a batch of SMILES to PubChem and return a CID per entry (same order).

    PubChem returns one CID per input SMILES in the same order, using 0 for
    compounds not found. Returns None when the request fails or the response
    length does not match the input — the caller should fall back to individual
    lookups in that case.

    Args:
        smiles_batch: SMILES strings to submit in one POST request.
        sleep_s: Delay after each attempt (seconds).
        retries: Extra retry attempts on transient server errors (429, 5xx).

    Returns:
        A CID (or None for "not found") per entry in ``smiles_batch``, same
        order; or None if the request failed or returned a mismatched count.
    """
    url = f"{PUBCHEM_BASE}/compound/smiles/cids/JSON"
    for attempt in range(retries + 1):
        try:
            r = requests.post(
                url,
                data={"smiles": "\n".join(smiles_batch)},
                timeout=(10, 60),
            )
            time.sleep(sleep_s)
            if r.status_code == 200:
                cids = r.json().get("IdentifierList", {}).get("CID", [])
                if len(cids) != len(smiles_batch):
                    return None  # count mismatch → caller falls back
                return [c if c != 0 else None for c in cids]
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(1.0 * (attempt + 1))
                continue
            return None
        except Exception:
            time.sleep(1.0 * (attempt + 1))
    return None


def pubchem_has_vendor(cid: int) -> int:
    """Return 1 if PubChem lists at least one commercial vendor for ``cid``, else 0.

    Fetches the PUG View "Chemical-Vendors" section and walks the response JSON
    recursively, looking for a "Chemical Vendors" TOC heading whose Information
    entries contain a Boolean True value.

    Args:
        cid: PubChem Compound ID to check.

    Returns:
        1 if a vendor is listed, else 0 (including on request/parse failure).
    """
    url = f"{PUBCHEM_VIEW}/data/compound/{cid}/JSON?heading=Chemical-Vendors"
    try:
        r = requests.get(url, timeout=(5, 15))
    except Exception:
        return 0
    if r.status_code != 200:
        return 0
    try:
        data = r.json()
    except Exception:
        return 0

    def _has_vendor(obj: Any) -> bool:
        """Recursively search the PUG View JSON tree for a vendor hit."""
        if isinstance(obj, dict):
            if obj.get("TOCHeading") == "Chemical Vendors":
                return any(
                    True in x.get("Value", {}).get("Boolean", [])
                    for x in obj.get("Information", [])
                )
            return any(_has_vendor(v) for v in obj.values())
        if isinstance(obj, list):
            return any(_has_vendor(x) for x in obj)
        return False

    return 1 if _has_vendor(data) else 0


# ── Resumable lookup helpers ──────────────────────────────────────────────────


def build_cid_map(
    smiles_list: List[str],
    cache_path: Path,
    sleep_s: float = 0.35,
    retries: int = 1,
    batch_size: int = 100,
) -> Dict[str, Optional[int]]:
    """Map SMILES → PubChem CID using batched POST requests, with CSV-based resumability.

    Sends up to ``batch_size`` SMILES per POST request (~100× fewer API calls
    than one-at-a-time). When a batch POST fails or returns a mismatched
    number of CIDs, each SMILES in that batch is retried individually. Results
    are flushed to ``cache_path`` after every batch so progress survives
    interruption.

    Args:
        smiles_list: Unique SMILES to look up (order preserved).
        cache_path: CSV with columns [cap_smiles, CID]; created if absent.
        sleep_s: Delay between API calls (seconds).
        retries: Retry attempts on transient server errors.
        batch_size: Number of SMILES per POST request (default 100).

    Returns:
        Mapping of SMILES → CID (int) or None if not found in PubChem.
    """
    cache: Dict[str, Optional[int]] = {}
    done: Set[str] = set()

    if cache_path.exists():
        old = pd.read_csv(cache_path)
        for _, row in old.iterrows():
            smi = str(row[SMILES_COL]).strip()
            done.add(smi)
            try:
                cid = int(float(row[CID_COL]))
                cache[smi] = cid if cid != 0 else None
            except Exception:
                cache[smi] = None
    else:
        pd.DataFrame(columns=[SMILES_COL, CID_COL]).to_csv(cache_path, index=False)

    todo = [s for s in smiles_list if s not in done]
    batches = [todo[i:i + batch_size] for i in range(0, len(todo), batch_size)]

    for batch in tqdm(batches, desc="CID lookup", unit="batch"):
        cids = _post_smiles_batch(batch, sleep_s=sleep_s, retries=retries)

        if cids is not None:
            pairs = list(zip(batch, cids))
        else:
            # Batch failed — fall back to individual lookups
            pairs = [
                (smi, pubchem_smiles_to_cid(smi, sleep_s=sleep_s, retries=retries))
                for smi in batch
            ]

        rows = []
        for smi, cid in pairs:
            cache[smi] = cid
            rows.append({SMILES_COL: smi, CID_COL: cid})

        chunk = pd.DataFrame(rows)
        chunk[CID_COL] = chunk[CID_COL].astype(pd.Int64Dtype())
        chunk.to_csv(cache_path, mode="a", header=False, index=False)

    return cache


def build_vendor_map(
    cid_list: List[int],
    cache_path: Path,
    sleep_s: float = 0.1,
) -> Dict[int, int]:
    """Map CID → vendor status (1/0), reading and writing a CSV cache for resumability.

    Skips CIDs already present in ``cache_path`` and flushes each result
    immediately so progress survives interruption.

    Args:
        cid_list: Unique CIDs to check.
        cache_path: CSV with columns [CID, vendor_status]; created if absent.
        sleep_s: Delay between API calls (seconds).

    Returns:
        Mapping of CID → 1 (vendor found) or 0 (not found / unavailable).
    """
    done: Dict[int, int] = {}

    if cache_path.exists():
        old = pd.read_csv(cache_path)
        for _, row in old.iterrows():
            try:
                done[int(float(row[CID_COL]))] = int(row["vendor_status"])
            except Exception:
                continue
    else:
        pd.DataFrame(columns=[CID_COL, "vendor_status"]).to_csv(cache_path, index=False)

    todo = [c for c in cid_list if c not in done]

    with cache_path.open("a", encoding="utf-8") as f:
        for cid in tqdm(todo, desc="Vendor lookup", unit="CID"):
            status = pubchem_has_vendor(cid)
            done[cid] = status
            f.write(f"{cid},{status}\n")
            f.flush()
            time.sleep(sleep_s)

    return done


# ── Main ──────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Populated ``argparse.Namespace``.
    """
    p = argparse.ArgumentParser(description="PubChem CID and vendor check for capped components.")
    p.add_argument("--input", type=Path, default=_PROJECT_ROOT / "data/processed/component_capped.csv",
                   help="Capped component CSV (capping_component output).")
    p.add_argument("--cid-cache", type=Path, default=_PROJECT_ROOT / "data/processed/component_smiles_to_cid.csv",
                   help="Resumable SMILES→CID cache CSV.")
    p.add_argument("--vendor-cache", type=Path, default=_PROJECT_ROOT / "data/processed/component_cid_to_vendor.csv",
                   help="Resumable CID→vendor cache CSV.")
    p.add_argument("--output", type=Path, default=_PROJECT_ROOT / "data/processed/component_check_cid_vendor.csv",
                   help="Final output CSV.")
    p.add_argument("--sleep", type=float, default=0.35,
                   help="Delay between PubChem API calls (seconds). Vendor calls use 30%% of this.")
    p.add_argument("--retries", type=int, default=1,
                   help="Retry attempts on transient server errors.")
    p.add_argument("--batch-size", type=int, default=100,
                   help="SMILES per batched POST request to PubChem (default 100).")
    return p.parse_args()


def main(
    input_path: Path,
    cid_cache: Path,
    vendor_cache: Path,
    output_path: Path,
    sleep_s: float = 0.35,
    retries: int = 1,
    batch_size: int = 100,
) -> None:
    """Look up PubChem CIDs and vendor status for every capped-component SMILES.

    Args:
        input_path: Capped component CSV (capping_component output).
        cid_cache: Resumable SMILES→CID cache CSV.
        vendor_cache: Resumable CID→vendor cache CSV.
        output_path: Final output CSV path.
        sleep_s: Delay between PubChem API calls (seconds).
        retries: Retry attempts on transient server errors.
        batch_size: SMILES per batched POST request to PubChem.
    """
    df = pd.read_csv(input_path)

    if "error" in df.columns:
        df = df[df["error"].isna()].copy()
    df = df.dropna(subset=[SMILES_COL]).drop_duplicates(subset=["component_id", SMILES_COL])
    print(f"Unique (component_id, smiles) pairs: {len(df)}")

    unique_smiles = df[SMILES_COL].astype(str).str.strip().unique().tolist()
    print(f"Unique SMILES to look up: {len(unique_smiles)}")

    cid_map = build_cid_map(unique_smiles, cid_cache, sleep_s=sleep_s, retries=retries, batch_size=batch_size)
    df[CID_COL] = df[SMILES_COL].map(cid_map).astype(pd.Int64Dtype())

    unique_cids = sorted({int(c) for c in df[CID_COL].dropna() if int(c) != 0})
    print(f"Unique CIDs found: {len(unique_cids)}")

    vendor_map = build_vendor_map(unique_cids, vendor_cache, sleep_s=sleep_s * 0.3)
    df["vendor_status"] = df[CID_COL].map(vendor_map)

    out = df[["component_id", SMILES_COL, CID_COL, "vendor_status"]]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)

    print(f"\nSaved → {output_path}  ({len(out)} rows)")
    print(f"  Vendor available   : {(out['vendor_status'] == 1).sum()}")
    print(f"  Vendor unavailable : {(out['vendor_status'] == 0).sum()}")
    print(f"  No CID found       : {out[CID_COL].isna().sum()}")


if __name__ == "__main__":
    args = parse_args()
    main(
        args.input,
        args.cid_cache,
        args.vendor_cache,
        args.output,
        sleep_s=args.sleep,
        retries=args.retries,
        batch_size=args.batch_size,
    )
