"""
fs_score.py
===========
FSscore (Focused Synthesizability Score; Neeser et al., 2024).
Raw: unbounded RankNet pairwise-comparison score (higher = easier) — NOT a
[0,1] value: verified against the pinned fsscore commit, whose Scorer.score()
returns the model's raw linear output with no sigmoid/clamp applied (the
sigmoid only appears in the *training* loss, over score differences).
Scaled: min-max normalized to [0,1] *within the scored batch*, since
fsscore's own absolute-calibration path (reverse_sigmoid against the
model's learned min/max) is commented out upstream and unavailable here.

Installed as a pip/git package (schwallergroup/fsscore). The checkpoint is
downloaded separately from figshare; path (FSSCORE_MODEL) comes from the
package __init__.

Device note
-----------
fsscore's Scorer builds a Lightning Trainer internally and defaults to
device='auto', which grabs a GPU if one is visible. On a login node that
has a GPU it cannot use, that crashes ("CUDA-capable device is busy or
unavailable"). Pass device='cpu' to force CPU, or run on a SLURM GPU node
where 'auto' works. Controlled here via the `device` argument (default
'auto' to match upstream behaviour on a proper compute node).
"""

import shutil
import tempfile

import numpy as np
import pandas as pd

COLUMNS = ["fs_score", "fs_score_scaled"]


def compute(smiles: list, device: str = "auto", batch_size: int = 128, num_workers: int = 4,) -> pd.DataFrame:
    print("Computing FSscore")
    from . import FSSCORE_MODEL  # lazy to avoid circular import
    from fsscore.models.ranknet import LitRankNet
    from fsscore.score import Scorer

    model = LitRankNet.load_from_checkpoint(str(FSSCORE_MODEL))
    # Scorer defaults to caching the batch's featurized graphs under its own
    # package install dir (PROCESSED_PATH), which inside the container sits
    # on the read-only SIF image -- writable only via `--writable-tmpfs`'s
    # RAM-backed overlay, whose capacity is capped by the SLURM job's memory
    # allocation, not real disk. A full-size batch (18k+ molecules) can
    # exhaust that overlay mid-write ([Errno 28] No space left on device)
    # even though the job has plenty of real disk. Point the cache at a
    # host-backed tempdir instead so it's bounded by real disk, not RAM.
    tmpdir = tempfile.mkdtemp(prefix="fsscore_")
    try:
        scorer = Scorer(
            model=model, device=device, batch_size=batch_size, num_workers=num_workers,
            graph_datapath=f"{tmpdir}/graphs_score.pt",
        )
        raw = np.array(scorer.score(smiles), dtype=np.float64)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    raw_range = np.nanmax(raw) - np.nanmin(raw)
    if raw_range > 0:
        scaled = (raw - np.nanmin(raw)) / raw_range
    else:
        scaled = np.full_like(raw, 0.5)  # degenerate batch: all-equal or all-NaN raw scores
    return pd.DataFrame({"fs_score": raw, "fs_score_scaled": scaled})
