"""Tests for the RetroTAC scoring server (`retrotac.serve`).

Focused on the three invariants a generative run depends on: one value per
submitted SMILES whatever happens to it, a load-time failure for anything
knowable at load time, and an endpoint shape that follows the model's own
output keys.
"""
import numpy as np
import pytest

from retrotac.serve.app import ScoringService, VALIDATION_SMILES
from retrotac.serve.schemas import EXAMPLE_SMILES
from conftest import FakeEnsemble

VALID = "COc1ccc(-c2ocnc2C(=O)NCCCc2cn(CCOCCOCCNc3cccc4c3C(=O)N(C3CCC(=O)NC3=O)C4=O)nn2)cc1I"
INVALID = "not-a-smiles((("


def test_startup_scores_a_reference_molecule():
    """Construction proves the whole featurize -> predict path works."""
    ensemble = FakeEnsemble()
    service = ScoringService(ensemble, source="tests/fake")
    assert ensemble.calls == [[VALIDATION_SMILES]]
    assert service.output_keys == ["mean", "std"]


def test_startup_rejects_a_width_mismatch():
    """A model returning the wrong number of values must not start listening."""
    with pytest.raises(RuntimeError, match="Start-up validation failed"):
        ScoringService(FakeEnsemble(n_values=3), source="tests/fake")


def test_startup_reports_a_predict_failure_with_a_diagnosis():
    """A load-time exception is re-raised naming the likely cause."""

    class Broken(FakeEnsemble):
        def predict(self, *args, **kwargs):
            raise ValueError("Feature shape mismatch, expected: 2265, got 1241")

    with pytest.raises(RuntimeError, match="fp_size/fp_radius"):
        ScoringService(Broken(), source="tests/fake")


def test_invalid_smiles_scores_null_in_place(service):
    """An unparseable SMILES is data: null in place, list length preserved."""
    response = service.predict([VALID, INVALID, VALID])
    assert [v is None for v in response["mean"]] == [False, True, False]
    assert len(response["mean"]) == len(response["std"]) == 3
    # The model never sees the unparseable one -- it would score an all-zero
    # fingerprint row rather than refuse it.
    assert service._ensemble.calls[-1] == [VALID, VALID]


def test_all_invalid_smiles_skips_the_model(service):
    """A batch with nothing scorable returns nulls without calling predict."""
    calls_before = len(service._ensemble.calls)
    response = service.predict([INVALID, INVALID])
    assert response == {"mean": [None, None], "std": [None, None]}
    assert len(service._ensemble.calls) == calls_before


def test_non_finite_output_becomes_null():
    """NaN is not valid JSON, so it is reported as null."""
    service = ScoringService(FakeEnsemble(), source="tests/fake")
    service._ensemble.non_finite = True
    response = service.predict([VALID, VALID])
    assert response["mean"][0] is None
    assert response["mean"][1] == pytest.approx(0.01 * len(VALID))


def test_misaligned_prediction_raises_rather_than_padding():
    """A caller cannot tell which molecule lost its score, so this must raise."""
    service = ScoringService(FakeEnsemble(), source="tests/fake")
    service._ensemble.n_values = 1
    with pytest.raises(RuntimeError, match="cannot be realigned"):
        service.predict([VALID, VALID])


def test_response_follows_the_models_output_keys():
    """A key added to the ensemble appears without a server change."""
    service = ScoringService(FakeEnsemble(keys=("mean", "std", "q_hat")), source="tests/fake")
    assert service.output_keys == ["mean", "std", "q_hat"]
    assert set(service.predict([VALID])) == {"mean", "std", "q_hat"}


def test_per_member_predictions_stay_off_the_wire(service):
    """The 27-array member block is dropped from the response."""
    assert "members" not in service.predict([VALID])


def test_health_does_no_model_work(client, service):
    """/health must stay cheap: it is what readiness polling hits."""
    calls_before = len(service._ensemble.calls)
    assert client.get("/health").json() == {"status": "ok"}
    assert len(service._ensemble.calls) == calls_before


def test_info_reports_enough_to_diagnose_a_mismatch(client):
    """/info names the model source, its members, and its feature config."""
    info = client.get("/info").json()
    assert info["source"] == "tests/fake"
    assert info["n_members"] == 3
    assert info["backends"] == {"gnn": 1, "mlp": 1, "xgb": 1}
    assert info["feature_config"]["fp_size"] == 2048
    assert info["output_keys"] == ["mean", "std"]


def test_predict_endpoint_returns_a_flat_keyed_payload(client):
    """The POST body is the documented example; the reply is flat JSON."""
    response = client.post("/predict", json={"smiles": [EXAMPLE_SMILES, INVALID]})
    assert response.status_code == 200
    payload = response.json()
    assert sorted(payload) == ["mean", "std"]
    assert payload["mean"][1] is None
    assert np.isfinite(payload["mean"][0])


def test_malformed_request_is_a_422_not_a_crash(client):
    """FastAPI's own validation handles a bad body, with detail in the reply."""
    response = client.post("/predict", json={"smiles": "CCO"})
    assert response.status_code == 422
    assert "detail" in response.json()


def test_swagger_examples_are_real_values(client):
    """Swagger shows `additionalProp1` for a bare mapping unless we set one."""
    schema = client.get("/openapi.json").json()["components"]["schemas"]
    assert schema["PredictResponse"]["example"] == {"mean": [0.83], "std": [0.07]}
    assert schema["PredictRequest"]["example"]["smiles"] == [EXAMPLE_SMILES]


def test_cleanup_runs_on_lifespan_shutdown(service):
    """Cleanup cannot live in a `finally` around uvicorn.

    uvicorn re-raises the signal it captured once shutdown completes, which
    terminates the process without unwinding `run_server`, so the port file
    would be left behind on every SIGTERM.
    """
    from fastapi.testclient import TestClient

    from retrotac.serve.app import create_app

    removed = []
    with TestClient(create_app(service, on_shutdown=lambda: removed.append(True))) as client:
        assert client.get("/health").status_code == 200
        assert not removed
    assert removed == [True]
