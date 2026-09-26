# RetroTAC: Learning Route-Derived Synthetic Accessibility for PROTACs

RetroTAC predicts the synthesizability of a PROTAC directly from its SMILES string, without
calling a retrosynthesis planner at inference time.

## 🔭 Overview

PROTACs are large by construction — a target-binding warhead and an E3-ligase ligand joined by a
linker — so whole-molecule accessibility scores tend to read their size as difficulty, while
retrosynthesis planners that would settle the question are too expensive to put inside a
generative or virtual-screening loop. RetroTAC reframes route assessment as single-molecule
property prediction.

The training label is a route-derived score that rewards short route depth and the balanced joining
of fragments, computed from routes found by the ShallowTree planner. Three architectures are
trained against it — XGBoost, an MLP, and a Chemprop GNN — and RetroTAC proper is a
Caruana-selected ensemble of 27 of their cross-validation models, reaching RMSE 0.132 and R² 0.647
on held-out molecules while reporting the spread across members as an uncertainty estimate. The
paper covers the score definition, data curation, leakage controls, and full results.

The repository holds two separately installed Python projects:

- **`retrotac/`** — the surrogate model: training, evaluation, and inference (`scripts/models/`).
- **`retro_scores/`** — the scoring code. `route_scores/` computes the route-derived training
  label; `mol_scores/` computes the six published molecule scores (SA, SC, RA, SYBA, GASA, FS)
  used as comparison baselines, not as training targets, and needs its own environment (see
  [retro_scores/README.md](retro_scores/README.md)).

## 🔁 Reproducibility

Every number, table, and figure in the paper is reproducible from this repository plus the
archives under [release/](release/), built by
[scripts/release/make_artifacts.py](scripts/release/make_artifacts.py):

| Archive | Size | Contents |
|---|---|---|
| `retrotac_results.tar.gz` | 11.7 MiB | Per-fold metrics, Optuna studies, ensemble results, paper figures |
| `retrotac_data.tar.gz` | 18.8 MiB | Pipeline inputs and every intermediate CSV, including the route-derived labels |
| `retrotac_checkpoints.tar.gz` | 3.4 MiB | The final XGBoost and MLP refits |
| `gnn_20260828_182305_final.ckpt.gz` | 90.7 MiB | The final GNN refit |

Start with `retrotac_results.tar.gz`. At under 12 MiB it carries every per-fold metric and Optuna
study, which is enough to re-derive the reported statistics without retraining anything or
downloading a single model weight. The four archives together are 124.5 MiB, and hold exactly the
three refits the paper reports — XGBoost, MLP, and GNN.

The [reproducibility guide](docs/README.md) then walks the full pipeline end to end — route and
molecule scoring, deduplication, scaffold-leakage analysis, cross-validated training, ensemble
selection, the DeepPSA comparison, and every figure — with the command for each step.

## ⚙️ Installation

RetroTAC targets Python 3.12 and manages dependencies with [uv](https://docs.astral.sh/uv/):

```bash
pip install uv
uv sync                     # inference only — covers most users
uv sync --extra training    # + training, tracking, plotting, PROTAC-Splitter
uv sync --extra scoring     # + Route scoring
```

Route-derived scoring needs only the base environment, since it is pure RDKit. The six molecule
scores are the exception: they need a separate, older stack that no uv extra provides — see
[retro_scores/README.md](retro_scores/README.md).

Where `uv sync` is impractical — for example, a shared cluster with a file-count quota — prebuilt
Apptainer containers cover the same profiles; see [apptainer/README.md](apptainer/README.md).
Full environment and cluster-specific notes live in [CONTRIBUTING.md](CONTRIBUTING.md).

## 🚀 Quickstart

The released archives carry the final GNN refit, the strongest single model in the paper (held-out
R² 0.615, RMSE 0.140). Unpack the checkpoint as described under
[Artifacts](docs/README.md#artifacts), then score SMILES through the Python API:

```python
from retrotac.models.gnn.model import CheMeleonRegressor

# Base path without the extension — load() appends ".ckpt" itself.
model = CheMeleonRegressor.load(
    "outputs/models/gnn_20260828_182305/gnn_20260828_182305_final", device="cpu"
)

smiles = ["COc1ccc(-c2ocnc2C(=O)NCCCc2cn(CCOCCOCCNc3cccc4c3C(=O)N(C3CCC(=O)NC3=O)C4=O)nn2)cc1I"]
preds = model.predict(smiles)   # shape (len(smiles), n_targets)
preds[:, 0]                     # the synthesizability column
```

Pass `device="cuda"` and a larger `batch_size` (256–512, watching GPU memory) to score a library
rather than a handful of molecules.

### Full ensemble

RetroTAC proper is the Caruana ensemble of 27 cross-validation members — 13 GNN, 7 MLP, 7 XGB —
which reaches RMSE 0.132 and R² 0.647, ahead of any single model, and reports the spread across
its members as an uncertainty estimate:

```python
from retrotac.models.ensemble import RetroTAC

model = RetroTAC.from_pretrained("retrotac_ensemble", device="cpu")
scores = model.predict(smiles)                      # weighted mean, shape (len(smiles),)

detailed = model.predict(smiles, return_details=True)
detailed["mean"]                                    # same weighted mean
detailed["std"]                                     # spread across members, as uncertainty
```

`from_pretrained` treats any path that exists on disk as a local model directory, so it needs no
network access. That directory holds `ensemble.json` plus the members under `members/`, staged
from a completed cross-validation run:

```bash
python scripts/models/push_to_hf.py \
    --weights outputs/results/results_20260828_182305/ensemble_weights_caruana.json \
    --cv-dir outputs/cv --local-dir retrotac_ensemble
```

> [!NOTE]
> The ensemble members are the per-fold models under `outputs/cv/`, which the released
> archives omit — the 27 selected members alone come to 1.4 GiB, dominated by 13 GNN checkpoints.
> Running the ensemble therefore means training the cross-validation models first, following
> [Training](docs/README.md#training). The trained ensemble will also be published to the Hugging
> Face Hub after peer review, at which point `from_pretrained` accepts a Hub repo id in place of
> the local path.

Also pass `n_jobs=-1` to parallelize featurization across members, and `load_strategy="lazy"` to
load members on demand instead of all at once.

## 📝 Notes on Implementation

Predictions are not clipped to [0, 1]. A value outside that range is a signal rather than a bug:
it means the molecule differs substantially from the training distribution.

The label's two endpoints are not symmetric. A molecule already in the purchasable stock scores 1,
whereas a search that fails to reach stock within the planner's budget scores 0 — which may record
genuine difficulty or merely a timeout or depth limit. Treat the continuous score and its ranking
as the primary output; the 0.7 threshold used for the classification metrics is a reporting
convention, not a calibrated decision boundary.

## 📚 Documentation

- [docs/README.md](docs/README.md) — reproducibility guide
- [CONTRIBUTING.md](CONTRIBUTING.md) — environment setup, coding standards, and the
  cross-validation scheme
- [apptainer/README.md](apptainer/README.md) — building and running the containers
- [retro_scores/README.md](retro_scores/README.md) — the scoring package's separate,
  older dependency stack

## 📄 License

MIT — see [LICENSE](LICENSE).
