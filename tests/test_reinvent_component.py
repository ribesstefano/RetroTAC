"""Tests for the REINVENT4 side, with no REINVENT install needed.

Everything here is verified against a two-entry namespace stub
(`tests/stub/reinvent_plugins/components/`) standing in for REINVENT's own
plugin tree: the component is imported the way REINVENT imports it, with the
stub and `reinvent/` on *separate* `sys.path` entries, which is the only way
the namespace-package merge under test actually gets exercised.

The subprocess tests exist because each needs its own interpreter state: a
namespace package's `__path__` is fixed at first import, so an `__init__.py`
experiment cannot be undone in-process.
"""
import subprocess
import sys
import tomllib
from dataclasses import fields

import numpy as np
import pytest

from conftest import REINVENT_DIR, STUB_DIR

PLUGIN_PATH = f"{STUB_DIR}{':'}{REINVENT_DIR}"


def run_python(code: str, pythonpath: str = PLUGIN_PATH) -> subprocess.CompletedProcess:
    """Run `code` in a fresh interpreter with `pythonpath` prepended.

    Args:
        code: Python source to run with `-c`.
        pythonpath: PYTHONPATH value for the child.

    Returns:
        The completed process, stdout/stderr captured as text.
    """
    env = {"PYTHONPATH": pythonpath, "PATH": "/usr/bin:/bin", "OMP_NUM_THREADS": "1",
           "PYTHONDONTWRITEBYTECODE": "1"}
    return subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env
    )


@pytest.fixture
def parameters():
    """The `Parameters` dataclass, imported through the namespace stub."""
    sys.path[:0] = [str(STUB_DIR), str(REINVENT_DIR)]
    try:
        from reinvent_plugins.components.comp_retrotac import Parameters

        return Parameters
    finally:
        del sys.path[:2]


@pytest.fixture
def component(monkeypatch):
    """A constructed component with the HTTP layer monkeypatched out.

    Returns a factory taking the `Parameters` the TOML would produce, and
    returning (component, recorded request paths).
    """
    sys.path[:0] = [str(STUB_DIR), str(REINVENT_DIR)]
    try:
        from reinvent_plugins.components import comp_retrotac
        from reinvent_plugins.components.client import RetroTACClient
    finally:
        del sys.path[:2]

    requests = []

    def fake_request(self, path, payload=None):
        requests.append(path)
        if path == "/health":
            return {"status": "ok"}
        n = len(payload["smiles"])
        return {"mean": [0.5] * n, "std": [0.1] * n}

    monkeypatch.setattr(RetroTACClient, "_request", fake_request)

    def build(params):
        return comp_retrotac.RetroTAC(params), requests

    build.module = comp_retrotac
    return build


# --- the namespace-package mechanics REINVENT's loader depends on -----------


def test_component_imports_through_a_merged_namespace_package():
    """Both path entries merge, so REINVENT's own modules stay importable."""
    result = run_python(
        "import reinvent_plugins.components as c;"
        "from reinvent_plugins.components import comp_retrotac;"
        "print(len(c.__path__), comp_retrotac.RetroTAC.__component)"
    )
    assert result.returncode == 0, result.stderr
    n_entries, tag = result.stdout.split()
    assert int(n_entries) == 2  # the stub's and ours, merged
    assert tag == "True"


def test_an_init_file_shadows_the_other_path_entry(tmp_path):
    """Confirm the negative: an `__init__.py` breaks the merge, so we ship none.

    Built as a throwaway two-entry tree rather than by writing into the real
    one, since the failure is exactly "the other directory's modules vanish".
    """
    mine, theirs = tmp_path / "mine", tmp_path / "theirs"
    for root, module in ((mine, "comp_mine.py"), (theirs, "add_tag.py")):
        package = root / "reinvent_plugins" / "components"
        package.mkdir(parents=True)
        (package / module).write_text("value = 1\n")

    code = "import reinvent_plugins.components.add_tag; print('merged')"
    assert run_python(code, f"{mine}:{theirs}").returncode == 0

    (mine / "reinvent_plugins" / "components" / "__init__.py").write_text("")
    shadowed = run_python(code, f"{mine}:{theirs}")
    assert shadowed.returncode != 0
    assert "No module named 'reinvent_plugins.components.add_tag'" in shadowed.stderr


def test_no_init_files_are_shipped():
    """The real plugin tree must stay a native namespace package."""
    assert not list(REINVENT_DIR.rglob("__init__.py"))


def test_only_the_component_file_is_named_comp():
    """The client must not be registered as a component, so it is not `comp_*`."""
    components = REINVENT_DIR / "reinvent_plugins" / "components"
    assert sorted(p.name for p in components.glob("comp_*.py")) == ["comp_retrotac.py"]
    assert (components / "client.py").exists()


def test_reinvents_plugin_discovery_finds_the_component():
    """Simulate REINVENT's walk_packages scan over the merged namespace."""
    result = run_python(
        "import pkgutil, importlib;"
        "import reinvent_plugins.components as c;"
        "found = [];"
        "[found.extend("
        "   name for name, obj in vars(importlib.import_module(m.name)).items()"
        "   if hasattr(obj, '__component')"
        ") for m in pkgutil.walk_packages(c.__path__, c.__name__ + '.')"
        " if m.name.rsplit('.', 1)[-1].startswith('comp_')];"
        "print(found)"
    )
    assert result.returncode == 0, result.stderr
    assert "RetroTAC" in result.stdout


# --- one request per population, not per endpoint ---------------------------


def test_two_endpoints_make_one_predict_call(component, parameters):
    """N endpoints must cost one request, else featurization is recomputed."""
    comp, requests = component(parameters(key=["mean", "std"], url=["http://127.0.0.1:9/"] * 2))
    results = comp(["CCO", "c1ccccc1", "CCN"])

    assert requests.count("/predict") == 1
    assert len(results.scores) == 2
    assert [list(s) for s in results.scores] == [[0.5] * 3, [0.1] * 3]


def test_readiness_is_checked_at_construction(component, parameters):
    """A config pointing at a dead server must stop the run before it samples."""
    comp, requests = component(parameters(key=["mean"]))
    assert requests == ["/health"]
    assert comp.keys == ["mean"]


def test_omitted_keys_fall_back_to_defaults(component, parameters):
    """A key absent from the TOML arrives short or None, never indexed blind."""
    comp, _ = component(parameters(key=[None, "std"], timeout=[120.0]))
    assert comp.keys == ["mean", "std"]
    assert comp.client.timeout == 120.0
    assert comp.fail_on_error is True


def test_conflicting_connection_settings_are_refused(component, parameters):
    """Two URLs would mean a request each, undoing the whole design."""
    with pytest.raises(ValueError, match="conflicting 'url'"):
        component(parameters(key=["mean", "std"],
                             url=["http://127.0.0.1:1", "http://127.0.0.1:2"]))


def test_unknown_output_key_names_what_the_server_has(component, parameters):
    """A typo'd `params.key` must say what the keys actually are."""
    comp, _ = component(parameters(key=["meen"]))
    with pytest.raises(KeyError, match="mean"):
        comp(["CCO"])


def test_fail_on_error_false_scores_nan_instead_of_raising(component, parameters, monkeypatch):
    """The opt-in path that survives a server restart mid-run."""
    comp, _ = component(parameters(key=["mean"], fail_on_error=[False]))
    monkeypatch.setattr(
        type(comp.client), "predict",
        lambda self, smiles: (_ for _ in ()).throw(RuntimeError("server gone")),
    )
    scores = comp(["CCO", "CCN"])
    assert np.isnan(scores.scores[0]).all()


# --- driven from the real TOML ---------------------------------------------


def test_scoring_toml_params_are_all_known_to_the_dataclass(parameters):
    """Every `params.*` key in the shipped TOML must exist on `Parameters`."""
    config = tomllib.loads((REINVENT_DIR / "scoring.toml").read_text())
    known = {f.name for f in fields(parameters)}

    endpoints = config["scoring"]["component"]
    retrotac = next(c["RetroTAC"] for c in endpoints if "RetroTAC" in c)
    used = {key for endpoint in retrotac["endpoint"] for key in endpoint.get("params", {})}
    assert used <= known, f"unknown params: {sorted(used - known)}"
    assert used, "the TOML sets no params at all -- it cannot be driving the component"


def test_scoring_toml_builds_the_component_the_way_reinvent_does(component, parameters):
    """Assemble `Parameters` as parallel lists from the TOML and run it."""
    config = tomllib.loads((REINVENT_DIR / "scoring.toml").read_text())
    retrotac = next(c["RetroTAC"] for c in config["scoring"]["component"] if "RetroTAC" in c)
    endpoints = retrotac["endpoint"]

    params = parameters(**{
        field.name: [endpoint.get("params", {}).get(field.name) for endpoint in endpoints]
        for field in fields(parameters)
    })
    comp, requests = component(params)
    results = comp(["CCO", "CCN"])

    assert comp.keys == ["mean", "std"]
    assert requests.count("/predict") == 1
    assert len(results.scores) == len(endpoints)


def test_uncertainty_endpoint_has_zero_weight():
    """A positive weight on `std` would optimise for familiarity, not quality."""
    config = tomllib.loads((REINVENT_DIR / "scoring.toml").read_text())
    retrotac = next(c["RetroTAC"] for c in config["scoring"]["component"] if "RetroTAC" in c)
    weights = {e["params"]["key"]: e["weight"] for e in retrotac["endpoint"]}
    assert weights["std"] == 0.0
    assert weights["mean"] > 0.0


# --- the client's own contract ---------------------------------------------


def test_client_chunks_above_max_batch_and_merges_back(component, parameters):
    """A population larger than max_batch costs several requests, one reply."""
    comp, requests = component(parameters(key=["mean"], max_batch=[2]))
    results = comp(["CCO"] * 5)
    assert requests.count("/predict") == 3
    assert len(results.scores[0]) == 5


def test_client_is_standard_library_only():
    """The point of the server is to add nothing to REINVENT's environment."""
    source = (REINVENT_DIR / "reinvent_plugins" / "components" / "client.py").read_text()
    for third_party in ("import numpy", "import requests", "import pandas", "import httpx"):
        assert third_party not in source


def test_client_runs_as_a_standalone_cli():
    """`python client.py --help` is the first debugging step; keep it working."""
    client_py = REINVENT_DIR / "reinvent_plugins" / "components" / "client.py"
    result = subprocess.run(
        [sys.executable, str(client_py), "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "--port-file" in result.stdout


def test_port_file_overrides_the_url_port(tmp_path, component, parameters):
    """`--port 0` plus a port file is what makes several servers per node work."""
    port_file = tmp_path / "retrotac.port"
    port_file.write_text("54321\n")
    comp, _ = component(parameters(key=["mean"], port_file=[str(port_file)]))
    assert comp.client.endpoint("/predict") == "http://127.0.0.1:54321/predict"
