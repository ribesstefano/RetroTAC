"""Score retrosynthesis routes (or route-free molecules) with an LLM judge via DSPy.
Defaults to Gemini; any litellm-style model string works, including OpenRouter
(`openrouter/<upstream-provider>/<model>`) -- see the `--model` CLI flag.

Pipeline
--------
    CSV row --parse--> ParsedRoute
        if resolved is True and route has 0 reactions -> skip the LLM,
                                             perfect score (target is in stock)
        elif resolved is False / no route  -> MoleculeScorer  (route-free)
        else                                -> RouteScorer     (grade the route;
                                             route-only by default, --score-steps
                                             also elicits per-step evaluations)
        --> whole-route (or whole-molecule), optionally per-step, evaluation
        --> flattened `llm_*` columns appended back onto the input CSV
        --> raw per-sample predictions persisted to JSONL.

Notes on the input
------------------
* The `score` column is IGNORED. It is not a reliable signal, so it is neither
  shown to the judge nor used anywhere.
* The routes are NOT from AiZynthFinder; there is no reaction-hash contract to
  honour. Steps are referenced by short human labels (S1, S2, ...) and, in the
  persisted output, by the reaction string itself.
* `resolved == False` means "ignore the route and judge the molecule's
  synthesizability without it".
* `resolved == True` with an empty route (`route == "{}"`, 0 reactions) means
  the target itself is a purchasable stock item -- nothing to disconnect. These
  rows never reach the LLM; see `ParsedRoute.in_stock` and `_in_stock_row`.

Design choices
--------------
* One LLM call scores the whole route and, if `--score-steps` is passed, every
  step too (RetrosynthesisRouteAndStepsScore vs. the route-only RetrosynthesisRouteScore).
  Route scoring is prioritized by default: route_score is the primary signal
  and skipping per-step output is cheaper/faster. Either way it's a single
  call: whole-route context is needed for the "protecting group" and
  "unnecessary step" categories, and it is far cheaper than N+1 calls per route.
* Structures are passed once via a compound legend (see route_parsing.render);
  reactions reference short tags, so a 300-character PROTAC SMILES is not
  repeated a dozen times. This keeps token cost bounded on huge molecules.
* Route mode emits `route_score` (0-100, higher = better); molecule mode emits
  `synthesizability_score` on the SAME 0-100, higher = easier orientation, so a
  single `llm_score` column is comparable across both modes.
* EnsembleScorer samples K times and aggregates to denoise the label and to
  expose an inter-sample agreement signal (a free quality check on the judge).
  K-sampling requires caching OFF (see build_lm) or the K calls collapse to one.

Signatures, output schemas, and dspy.Module scorers live in `models.py`; this
file is LM configuration, the batch runner, and the CLI. The PROTAC/non-PROTAC
classifier (`models.ProtacClassifier`) is a separate, opt-in judge run via
`classify_protac.py`, not part of this pipeline.

The surrogate model is handled separately (not in this file).
"""

from __future__ import annotations

import json
import os
import sys
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import mean
from typing import Optional

import dspy
from dotenv import load_dotenv
from tqdm import tqdm

from models import USE_COT_DEFAULT, EnsembleScorer

# API keys (GEMINI_API_KEY, OPENROUTER_API_KEY) live in .env at the project root;
# load them into the environment before JUDGE_MODEL/os.getenv reads below.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_PROJECT_ROOT / ".env")

# Add src/ to the path so protac_synth.route_parsing is importable.
sys.path.append(str(_PROJECT_ROOT / "src"))

from scripts.llm_scoring.route_parsing import ParsedRoute, describe_molecule, parse_route_row  # noqa: E402

# --------------------------------------------------------------------------- #
# 1. LM configuration
# --------------------------------------------------------------------------- #
# Model strings follow litellm's `<provider>/<model>` convention (DSPy's LM is a
# thin litellm wrapper):
#   - Gemini via Google AI Studio API key uses the `gemini/` prefix (Vertex uses
#     `vertex_ai/` + GCP creds). Preview strings rotate -- check the active model
#     list before a big run. As of mid-2026: gemini-3.1-pro-preview (best
#     reasoning, ~$2/$12 per 1M tok), gemini-3-flash-preview (cheap bulk, ~$0.5/$3).
#   - OpenRouter uses `openrouter/<upstream-provider>/<model>`, e.g.
#     `openrouter/google/gemini-3.1-pro-preview` or `openrouter/anthropic/claude-...`.
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gemini/gemini-3.1-pro-preview")


def _resolve_api_key(model: str) -> Optional[str]:
    """Pick the API key matching the model's provider prefix. OpenRouter proxies
    every upstream provider behind one key, so it must be checked before the
    upstream provider name (e.g. `openrouter/google/...` is NOT a `gemini/` call)."""
    if model.startswith("openrouter/"):
        return os.getenv("OPENROUTER_API_KEY")
    if model.startswith("gemini/"):
        return os.getenv("GEMINI_API_KEY")
    return None  # let litellm fall back to the provider's own env var


def build_lm(model: str = JUDGE_MODEL, temperature: Optional[float] = None,
             max_tokens: int = 16000) -> dspy.LM:
    """Construct a judge LM. cache=False is REQUIRED for ensemble sampling:
    with caching on, K identical requests return one cached completion and the
    ensemble variance collapses to zero. temperature=None leaves it unset, so
    dspy.LM/litellm fall back to the chosen model's own default."""
    return dspy.LM(model, temperature=temperature, max_tokens=max_tokens,
                   api_key=_resolve_api_key(model), cache=False)


def configure_judge(model: str = JUDGE_MODEL, temperature: Optional[float] = None,
                    max_tokens: int = 16000) -> dspy.LM:
    lm = build_lm(model, temperature, max_tokens)
    # JSONAdapter makes typed/structured outputs robust across providers.
    dspy.configure(lm=lm, adapter=dspy.JSONAdapter())
    return lm


@contextmanager
def _use_lm(lm: dspy.LM):
    """Thread-local LM override so each worker reads its own call history."""
    ctx = getattr(dspy, "context", None) or dspy.settings.context
    with ctx(lm=lm):
        yield


# --------------------------------------------------------------------------- #
# 2. Flattening + serialization
# --------------------------------------------------------------------------- #
def to_output_row(pred: dspy.Prediction) -> dict:
    """Flatten an EnsembleScorer prediction into `llm_*` columns to append to the
    input CSV. `llm_score` is the primary 0-100 (higher=better) value for BOTH
    modes; route-only fields are null in molecule mode and vice versa."""
    row = {
        "llm_mode": pred.mode,
        "llm_score": pred.score_mean,
        "llm_score_median": pred.score_median,
        "llm_score_std": pred.score_std,            # label-noise proxy (0 if k=1)
        "llm_category": pred.category,
        "llm_category_agreement": pred.category_agreement,
        "llm_confidence": pred.confidence_mean,
        "llm_n_samples": pred.n_samples,
        "llm_risk_factors": "; ".join(pred.risk_factors) if pred.risk_factors else "",
        "llm_rationale": pred.rationale,
        "llm_route_description": pred.route_description,  # legend+steps the rationale's [I2]-style tags refer to
        "llm_n_building_blocks": len(pred.parsed.building_blocks),
        "llm_resolved_depth": pred.parsed.resolved_depth,
    }
    if pred.mode == "route":
        steps = pred.step_score_mean or {}
        row.update({
            "llm_n_steps": len(pred.parsed.reactions),
            "llm_mean_step_score": (mean(steps.values()) if steps else None),
            "llm_min_step_score": (min(steps.values()) if steps else None),  # weakest link
            "llm_step_scores": json.dumps({k: round(v, 2) for k, v in steps.items()}) if steps else "",
        })
    else:
        row.update({
            "llm_n_steps": 0,
            "llm_mean_step_score": None,
            "llm_min_step_score": None,
            "llm_step_scores": "",
        })
    return row


def _in_stock_row(parsed: ParsedRoute) -> dict:
    """Perfect-score row for `parsed.in_stock` targets: resolved with 0
    reactions means the target IS the stock item, so there is nothing to
    judge. Mirrors the `llm_*` key set of `to_output_row` so both paths
    concatenate onto the same CSV columns."""
    return {
        "llm_mode": "in_stock",
        "llm_score": 100.0,
        "llm_score_median": 100.0,
        "llm_score_std": 0.0,
        "llm_category": "Readily synthesizable (standard building blocks & couplings)",
        "llm_category_agreement": 1.0,
        "llm_confidence": 100.0,
        "llm_n_samples": 0,
        "llm_risk_factors": "",
        "llm_rationale": "Resolved with 0 disconnections: the target is itself a "
                         "purchasable stock item. Not sent to the LLM judge.",
        "llm_route_description": "",
        "llm_n_building_blocks": len(parsed.building_blocks),
        "llm_resolved_depth": parsed.resolved_depth,
        "llm_n_steps": 0,
        "llm_mean_step_score": None,
        "llm_min_step_score": None,
        "llm_step_scores": "",
    }


def _dump_samples(pred: dspy.Prediction) -> list[dict]:
    """Per-sample structured predictions -> plain dicts for JSONL persistence.
    These are the extracted LLM predictions; targets can be re-derived from them
    without re-querying Gemini."""
    out = []
    for s in pred.samples:
        if hasattr(s, "route_evaluation"):
            out.append({
                "mode": "route",
                "route_evaluation": s.route_evaluation.model_dump(),
                "step_evaluations": [se.model_dump() for se in s.step_evaluations],
            })
        elif hasattr(s, "molecule_evaluation"):
            out.append({
                "mode": "molecule_only",
                "molecule_evaluation": s.molecule_evaluation.model_dump(),
            })
    return out


def _route_reactions_dump(parsed: ParsedRoute) -> list[dict]:
    """Reaction table so step feedback (keyed by S-label) is joinable without a hash."""
    return [
        {"label": r.label, "depth": r.depth,
         "product": r.product, "precursors": r.precursors, "note": r.metadata}
        for r in parsed.reactions
    ]


def _response_text(hist_entry: dict) -> str:
    o = hist_entry.get("outputs")
    if isinstance(o, list):
        return "\n".join(str(x) for x in o)
    if o is not None:
        return str(o)
    r = hist_entry.get("response")
    return str(r) if r is not None else ""


def _extract_raw(lm: dspy.LM, n: int) -> list[dict]:
    """Best-effort capture of the last n raw completions from this worker's LM.
    Thread-safe because each worker owns its own LM instance (see _use_lm)."""
    hist = getattr(lm, "history", None) or []
    out = []
    for h in hist[-n:]:
        try:
            out.append({
                "response_text": _response_text(h),
                "usage": h.get("usage"),
                "cost": h.get("cost"),
            })
        except Exception:  # pragma: no cover
            pass
    return out


# --------------------------------------------------------------------------- #
# 3. Batch runner
# --------------------------------------------------------------------------- #
def score_routes_parallel(rows: list[dict], scorer: Optional[dspy.Module] = None,
                          k: int = 3, num_threads: int = 8,
                          use_cot: bool = USE_COT_DEFAULT,
                          score_steps: bool = False,
                          model: str = JUDGE_MODEL,
                          temperature: Optional[float] = None,
                          capture_raw: bool = True):
    """Score many CSV rows concurrently.

    Returns (out_rows, raw_records, errors):
      * out_rows     : one dict per row (row_index + llm_* cols; llm_error set on failure)
      * raw_records  : one dict per row for JSONL (structured samples + raw completions)
      * errors       : subset that raised, for the console summary
    Every input row yields exactly one out_row so the CSV append stays aligned."""
    scorer = scorer or EnsembleScorer(k=k, use_cot=use_cot, score_steps=score_steps)

    def _one(i: int, row: dict):
        try:
            parsed = parse_route_row(row)
            if parsed.in_stock:  # purchasable as-is -- no LLM call needed
                out = _in_stock_row(parsed)
                out_row = {"row_index": i, "llm_error": None, **out}
                raw_rec = {
                    "row_index": i,
                    "smiles": row.get("SMILES"),
                    "mode": "in_stock",
                    "resolved": parsed.resolved,
                    "n_samples": 0,
                    "aggregate": out,
                    "samples": [],
                    "raw_responses": [],
                    "error": None,
                    "building_blocks": parsed.building_blocks,
                }
                return out_row, raw_rec, None
            lm = build_lm(model, temperature)
            with _use_lm(lm):
                pred = scorer(parsed)
            out = to_output_row(pred)
            out_row = {"row_index": i, "llm_error": None, **out}
            raw_rec = {
                "row_index": i,
                "smiles": row.get("SMILES"),
                "mode": pred.mode,
                "resolved": parsed.resolved,
                "n_samples": pred.n_samples,
                "aggregate": out,
                "samples": _dump_samples(pred),
                "raw_responses": _extract_raw(lm, pred.n_samples) if capture_raw else [],
                "error": None,
            }
            if pred.mode == "route":
                raw_rec["reactions"] = _route_reactions_dump(parsed)
            else:
                raw_rec["molecular_context"] = describe_molecule(parsed.target)
            return out_row, raw_rec, None
        except Exception as e:
            err = {"row_index": i, "smiles": row.get("SMILES"), "error": repr(e)}
            out_row = {"row_index": i, "llm_mode": None, "llm_score": None,
                       "llm_error": repr(e)}
            raw_rec = {"row_index": i, "smiles": row.get("SMILES"),
                       "samples": [], "raw_responses": [], "error": repr(e)}
            return out_row, raw_rec, err

    out_rows, raw_records, errors = [], [], []
    with ThreadPoolExecutor(max_workers=num_threads) as ex:
        futs = [ex.submit(_one, i, r) for i, r in enumerate(rows)]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="scoring routes"):
            o, raw, err = fut.result()
            out_rows.append(o)
            raw_records.append(raw)
            if err:
                errors.append(err)
    return out_rows, raw_records, errors


def score_csv(in_path: str, out_path: str, k: int = 3, num_threads: int = 8,
              limit: Optional[int] = None, use_cot: bool = USE_COT_DEFAULT,
              score_steps: bool = False,
              model: str = JUDGE_MODEL, temperature: Optional[float] = None,
              raw_jsonl: Optional[str] = None) -> None:
    """Score every row and write `<input columns> + llm_* columns` to out_path.
    If raw_jsonl is given, also persist raw per-sample predictions there."""
    import ast
    import pandas as pd

    df = pd.read_csv(in_path)
    if limit:
        df = df.sample(frac=1).head(limit)
    rows = df.to_dict("records")
    for r in rows:  # BBs stored as a str repr in the CSV
        if isinstance(r.get("BBs"), str):
            try:
                r["BBs"] = ast.literal_eval(r["BBs"])
            except (ValueError, SyntaxError):
                pass

    configure_judge(model=model, temperature=temperature)  # sets JSONAdapter globally; per-row LMs are built in workers
    scorer = EnsembleScorer(k=k, use_cot=use_cot, score_steps=score_steps)
    out_rows, raw_records, errors = score_routes_parallel(
        rows, scorer=scorer, num_threads=num_threads, model=model, temperature=temperature,
        capture_raw=(raw_jsonl is not None))

    # Append llm_* columns back onto the original frame, aligned by row_index.
    llm_df = (pd.DataFrame(out_rows)
              .set_index("row_index")
              .reindex(range(len(df))))
    out_df = pd.concat([df.reset_index(drop=True),
                        llm_df.reset_index(drop=True)], axis=1)
    out_df.to_csv(out_path, index=False)

    n_ok = int(out_df["llm_score"].notna().sum()) if "llm_score" in out_df else 0
    n_mol = int((out_df.get("llm_mode") == "molecule_only").sum()) if "llm_mode" in out_df else 0
    n_stock = int((out_df.get("llm_mode") == "in_stock").sum()) if "llm_mode" in out_df else 0
    print(f"appended predictions for {n_ok}/{len(df)} rows "
          f"({n_mol} scored route-free, {n_stock} skipped as in-stock) -> {out_path};  {len(errors)} errors")

    if raw_jsonl:
        raw_records.sort(key=lambda d: d["row_index"])
        with open(raw_jsonl, "w") as fh:
            for rec in raw_records:
                fh.write(json.dumps(rec, default=str) + "\n")
        print(f"raw predictions -> {raw_jsonl}")

    if errors:
        err_path = out_path.replace(".csv", "_errors.csv")
        pd.DataFrame(errors).to_csv(err_path, index=False)
        print(f"errors -> {err_path}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("in_csv")
    ap.add_argument("out_csv", help="input columns + appended llm_* columns")
    ap.add_argument("--k", type=int, default=3,
                    help="LLM samples per row (denoising). "
                         "For each row, the LLM will be called K times and the results aggregated")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="Get score for the first N rows (for testing)")
    ap.add_argument("--cot", dest="use_cot", action="store_true",
                    help="use ChainOfThought (zero-shot reasoning field); default off")
    ap.add_argument("--score-steps", dest="score_steps", action="store_true",
                    help="also elicit per-step StepEvaluations alongside the route score "
                         "(RetrosynthesisRouteAndStepsScore). Default is route-only "
                         "(RetrosynthesisRouteScore): route_score is prioritized as the "
                         "primary signal, and skipping step evaluations is cheaper/faster.")
    ap.add_argument("--model", default=JUDGE_MODEL,
                    help="litellm-style model string, '<provider>/<model>'. "
                         "Direct Gemini: 'gemini/gemini-3.1-pro-preview' (needs "
                         "GEMINI_API_KEY). Via OpenRouter: "
                         "'openrouter/<upstream-provider>/<model>', e.g. "
                         "'openrouter/google/gemini-3.1-pro-preview' (needs "
                         "OPENROUTER_API_KEY). Default: %(default)s")
    ap.add_argument("--temperature", type=float, default=None,
                    help="sampling temperature. Default: unset, i.e. dspy.LM/litellm "
                         "fall back to the chosen model's own default.")
    ap.add_argument("--raw-jsonl", default=None,
                    help="where to save raw per-sample predictions "
                         "(default: <out_csv>_raw.jsonl; pass the string 'none' to disable)")
    a = ap.parse_args()

    raw = a.raw_jsonl
    if raw is None:
        raw = a.out_csv.rsplit(".", 1)[0] + "_raw.jsonl"
    elif raw.lower() == "none":
        raw = None

    score_csv(a.in_csv, a.out_csv, k=a.k, num_threads=a.threads,
              limit=a.limit, use_cot=a.use_cot, score_steps=a.score_steps,
              model=a.model, temperature=a.temperature, raw_jsonl=raw)
