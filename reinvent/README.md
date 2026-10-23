# Scoring PROTACs with RetroTAC inside REINVENT4

RetroTAC runs as a small HTTP server in **its own environment**, and REINVENT4 reaches it through
a standard-library scoring component in this directory. Nothing is installed into REINVENT, and
nothing from REINVENT is installed here.

Two reasons for the split, in order of importance:

1. **Dependency isolation.** REINVENT4 pins its own `torch` and `rdkit`. RetroTAC's ensemble holds
   both `torch` (13 Chemprop GNNs, 7 MLPs) and `xgboost` (7 boosters), and those two bundle
   separate `libomp` copies that **segfault** when both run multi-threaded in one process. Separate
   processes make the collision impossible; the server sets `OMP_NUM_THREADS=1` before either
   import regardless.
2. **A warm model.** The 27 members are 1.4 GiB and take seconds to deserialize. REINVENT's
   built-in external-process component would spawn a process per scoring call and pay that cost
   thousands of times per run. The server pays it once.

## 1. Start the server (RetroTAC environment)

```bash
uv sync --extra serve
python -m retrotac.serve                      # http://127.0.0.1:8000, docs at /docs
```

It binds loopback only — there is no authentication, so `--host 0.0.0.0` on a shared node would
expose inference to everyone on the network. The ensemble is pulled from the Hub
(`ailab-bio/RetroTAC`) on first use and cached under `~/.cache/huggingface/`; pass `--model
/path/to/retrotac_ensemble` to serve a local directory instead, which needs no network at all.

Useful flags:

| Flag | Why |
|---|---|
| `--port 0 --port-file /tmp/retrotac.port` | Let the kernel pick the port and write it out, so several scoring jobs can share one node. The file is removed on shutdown. |
| `--load-strategy lazy --max-resident 6` | Hold only a few members in memory instead of all 27, at the cost of reloading. |
| `--device cuda --gnn-batch-size 512` | Put the GNN members on a GPU and feed them larger batches. |
| `--progress` | Restore the per-request `tqdm` bars, which are suppressed by default. |

The server **refuses to start** if the ensemble cannot score a reference molecule: a feature-width
or missing-artifact problem would otherwise surface as a 500 on every request for the life of the
process. Three endpoints:

- `GET /health` — liveness, no model work.
- `GET /info` — model source, member counts per backend, feature config, output keys.
- `POST /predict` — `{"smiles": [...]}` in, `{"mean": [...], "std": [...]}` out.

The response mirrors `RetroTAC.predict(return_details=True)` key for key, so a client that knows
the Python API knows the wire format: `mean` is the Caruana-weighted synthesizability prediction,
`std` the spread across members. One value per submitted SMILES, always — an unparseable SMILES
comes back as `null` in place rather than failing the batch, since a 500 would kill a generative
run at step 1400 over one malformed string.

## 2. Check it from the client (no REINVENT needed)

[`reinvent_plugins/components/client.py`](reinvent_plugins/components/client.py) is
standard-library-only and runs standalone. This is what to debug with:

```bash
cd reinvent/reinvent_plugins/components
python client.py --info
python client.py --smiles "CCO" "not-a-smiles(((" --key mean
python client.py --input molecules.smi --port-file /tmp/retrotac.port --key mean
```

## 3. Point REINVENT at it

`PYTHONPATH` must name the directory **containing** `reinvent_plugins` — this one — not
`components/`:

```bash
cd reinvent
PYTHONPATH=$PWD reinvent -l scoring.log scoring.toml
```

[`scoring.toml`](scoring.toml) scores the `mean` endpoint through a sigmoid transform, logs `std`
at **zero weight**, and bounds MW and SlogP to the PROTAC range. Three notes on it:

- The transform window is where the objective actually lives, and the values shipped are
  **placeholders**. RetroTAC's label runs from 0 (the planner never reached purchasable stock
  within its budget — which may be real difficulty or merely a timeout) to 1 (the molecule is
  itself in stock), and the paper's 0.7 is a reporting convention, not a calibrated boundary. Set
  the window from your own thresholds.
- Keep `std` at weight 0. Logged, rising uncertainty is the earliest sign a run has drifted out of
  RetroTAC's training domain. Given a positive weight it would instead teach the optimiser to
  propose molecules near the training set rather than synthesizable ones.
- The property bounds are not decoration: without them a run drifts to molecules RetroTAC has no
  basis for scoring.

All endpoints of one component are served by **one** request per population, with `key` varying
per endpoint and the connection settings (`url`, `port_file`, `timeout`, `max_batch`) shared. That
is why conflicting connection values across endpoints are an error rather than silently honoured —
honouring them would mean a request each, re-featurizing the same molecules. A second server needs
a second `[scoring.component.RetroTAC]` block.

`params.fail_on_error` defaults to `true`. Setting it `false` scores a batch `nan` and carries on,
which survives a server restart mid-run but silently wastes every remaining step if the server
stays down.

## What is verified, and what is not

Run the suite from the repository root (`uv sync --extra serve --extra dev`):

```bash
OMP_NUM_THREADS=1 pytest
```

[`tests/test_serve.py`](../tests/test_serve.py) covers the server over a fake ensemble: alignment,
`null` for unparseable SMILES and non-finite outputs, the start-up validation, the lifespan
cleanup, and the OpenAPI examples. [`tests/test_reinvent_component.py`](../tests/test_reinvent_component.py)
covers this directory by importing the component the way REINVENT does, through a two-entry
namespace stub under `tests/stub/`: the namespace merge and its negative (an `__init__.py` here
shadows REINVENT's own package — that is why there are none), REINVENT's `walk_packages`
discovery, one `/predict` call for N endpoints, chunking, the port-file override, and the shipped
TOML driving the component with `Parameters` assembled as parallel lists.

> [!IMPORTANT]
> **Unverified:** everything REINVENT itself owns. This repository has no REINVENT install, so
> `run_type`, the `[scoring]`/`[[stage]]` layout, the `transform.*` coefficients, `termination`,
> and the built-in `MolecularWeight`/`SlogP` components in `scoring.toml` are written from
> REINVENT4's documentation and have never been run. The RetroTAC component block, its `params`
> keys, the client, and the server are the tested half. Check `scoring.toml` against your own
> REINVENT version before trusting a run's output.
