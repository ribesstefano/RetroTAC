"""Classify molecules as PROTAC / non-PROTAC with an LLM judge via DSPy.

Standalone from `llm_scoring.py`: takes any CSV with a SMILES column (name
configurable via `--smiles-column`, default "SMILES") and appends
`llm_protac_*` columns. No route / AiZynthFinder columns are required -- this
is a per-molecule classification, not a synthesizability score.

Reuses `llm_scoring.py`'s LM configuration (build_lm/configure_judge/_use_lm,
same `--model`/`--temperature` conventions) but calls `models.ProtacClassifier`,
a dedicated module distinct from the route/molecule scoring modules.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm

from llm_scoring import JUDGE_MODEL, _use_lm, build_lm, configure_judge
from models import ProtacClassifier


def classify_smiles_parallel(smiles_list: list, include_rationale: bool = False,
                             num_threads: int = 8, model: str = JUDGE_MODEL,
                             temperature: Optional[float] = None) -> list[dict]:
    """Classify many SMILES concurrently.

    Returns one dict per input SMILES, keyed by `row_index` (its position in
    `smiles_list`), so the caller can align results back onto the input CSV
    even if some calls raise.

    Args:
        smiles_list: Molecule SMILES to classify, in input order.
        include_rationale: Also elicit a concise rationale string (extra tokens).
        num_threads: Worker threads for concurrent LLM calls.
        model: litellm-style model string.
        temperature: Sampling temperature; None leaves the model's own default.

    Returns:
        One dict per input SMILES with `llm_protac_*` keys (`llm_protac_error`
        set and the rest None on failure).
    """
    classifier = ProtacClassifier(include_rationale=include_rationale)

    def _one(i: int, smiles: str) -> dict:
        try:
            lm = build_lm(model, temperature)
            with _use_lm(lm):
                pred = classifier(smiles)
            c = pred.classification
            return {
                "row_index": i,
                "llm_protac_label": c.label,
                "llm_protac_confidence": c.confidence,
                "llm_protac_rationale": getattr(c, "rationale", ""),
                "llm_protac_error": None,
            }
        except Exception as e:
            return {
                "row_index": i,
                "llm_protac_label": None,
                "llm_protac_confidence": None,
                "llm_protac_rationale": "",
                "llm_protac_error": repr(e),
            }

    out_rows = []
    with ThreadPoolExecutor(max_workers=num_threads) as ex:
        futs = [ex.submit(_one, i, s) for i, s in enumerate(smiles_list)]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="classifying PROTAC/non-PROTAC"):
            out_rows.append(fut.result())
    return out_rows


def classify_csv(in_path: Path, out_path: Path, smiles_column: str = "SMILES",
                 include_rationale: bool = False, num_threads: int = 8,
                 limit: Optional[int] = None, model: str = JUDGE_MODEL,
                 temperature: Optional[float] = None) -> None:
    """Append `llm_protac_*` columns to every row of `in_path` and write `out_path`.

    Args:
        in_path: Input CSV containing at least `smiles_column`.
        out_path: Output CSV path (input columns + appended `llm_protac_*`).
        smiles_column: Name of the SMILES column in `in_path`.
        include_rationale: Also elicit a concise rationale string per molecule.
        num_threads: Worker threads for concurrent LLM calls.
        limit: If set, classify only the first N rows (for testing).
        model: litellm-style model string.
        temperature: Sampling temperature; None leaves the model's own default.
    """
    df = pd.read_csv(in_path)
    if smiles_column not in df.columns:
        raise ValueError(f"column {smiles_column!r} not found in {in_path}; "
                         f"available columns: {list(df.columns)}")
    if limit:
        df = df.head(limit)

    configure_judge(model=model, temperature=temperature)  # sets JSONAdapter globally; per-row LMs are built in workers
    out_rows = classify_smiles_parallel(
        df[smiles_column].tolist(), include_rationale=include_rationale,
        num_threads=num_threads, model=model, temperature=temperature)

    # Append llm_protac_* columns back onto the original frame, aligned by row_index.
    llm_df = (pd.DataFrame(out_rows)
              .set_index("row_index")
              .reindex(range(len(df))))
    out_df = pd.concat([df.reset_index(drop=True),
                        llm_df.reset_index(drop=True)], axis=1)
    out_df.to_csv(out_path, index=False)

    n_ok = int(out_df["llm_protac_label"].notna().sum())
    n_protac = int((out_df["llm_protac_label"] == "PROTAC").sum())
    n_err = int(out_df["llm_protac_error"].notna().sum())
    print(f"classified {n_ok}/{len(df)} molecules ({n_protac} PROTAC) -> {out_path};  {n_err} errors")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("in_csv", type=Path)
    ap.add_argument("out_csv", type=Path, help="input columns + appended llm_protac_* columns")
    ap.add_argument("--smiles-column", default="SMILES", help="name of the SMILES column in in_csv")
    ap.add_argument("--rationale", dest="include_rationale", action="store_true",
                    help="also elicit a concise rationale string (extra tokens); default off")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="classify only the first N rows (for testing)")
    ap.add_argument("--model", default=JUDGE_MODEL,
                    help="litellm-style model string, '<provider>/<model>'. See llm_scoring.py --model. "
                         "Default: %(default)s")
    ap.add_argument("--temperature", type=float, default=None,
                    help="sampling temperature. Default: unset, i.e. dspy.LM/litellm "
                         "fall back to the chosen model's own default.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    classify_csv(args.in_csv, args.out_csv, smiles_column=args.smiles_column,
                include_rationale=args.include_rationale, num_threads=args.threads,
                limit=args.limit, model=args.model, temperature=args.temperature)


if __name__ == "__main__":
    main()
