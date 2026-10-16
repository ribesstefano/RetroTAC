"""Request/response schemas for the RetroTAC scoring server.

The response deliberately mirrors `RetroTAC.predict(return_details=True)`
key-for-key (minus the per-member block, which is 27 arrays wide): one JSON
object whose keys are the model's own output keys -- "mean" and "std" today --
each holding one value per submitted SMILES. A key added to the ensemble
therefore shows up in the payload without a change here, and a client that
knows the Python API already knows the wire format.

Every field carries an explicit `example`: Swagger renders
`additionalProp1`/`"string"` placeholders for a bare mapping, since it has no
field names to show, and the examples below are what make "Try it out" at
`/docs` succeed unmodified.
"""
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, RootModel

EXAMPLE_SMILES = (
    "COc1ccc(-c2ocnc2C(=O)NCCCc2cn(CCOCCOCCNc3cccc4c3C(=O)N(C3CCC(=O)NC3=O)C4=O)nn2)cc1I"
)


class PredictRequest(BaseModel):
    """A batch of SMILES to score."""

    model_config = ConfigDict(json_schema_extra={"example": {"smiles": [EXAMPLE_SMILES]}})

    smiles: List[str] = Field(..., description="SMILES strings to score, in order.")


class PredictResponse(RootModel[Dict[str, List[Optional[float]]]]):
    """The ensemble's own output keys, each aligned with the request's SMILES.

    A `null` entry means the molecule was not scorable -- an unparseable
    SMILES, or a non-finite model output, which is not representable in JSON.
    List length always equals the number of submitted SMILES, so a caller can
    index straight back into its own batch.
    """

    model_config = ConfigDict(
        json_schema_extra={"example": {"mean": [0.83], "std": [0.07]}}
    )


class InfoResponse(BaseModel):
    """What this process has loaded -- enough to diagnose a mismatch."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "source": "ailab-bio/RetroTAC",
                "retrotac_version": "0.1.0",
                "n_members": 27,
                "backends": {"gnn": 13, "mlp": 7, "xgb": 7},
                "feature_config": {
                    "fp_size": 2048,
                    "fp_radius": 3,
                    "use_fingerprints": True,
                    "use_descriptors": True,
                },
                "output_keys": ["mean", "std"],
                "device": "cpu",
                "load_strategy": "eager",
            }
        }
    )

    source: str = Field(..., description="Hub repo id or local directory the ensemble came from.")
    retrotac_version: str = Field(..., description="Installed retrotac package version.")
    n_members: int = Field(..., description="Number of ensemble members loaded.")
    backends: Dict[str, int] = Field(..., description="Member count per backend kind.")
    feature_config: Dict[str, object] = Field(..., description="config.json of the ensemble.")
    output_keys: List[str] = Field(..., description="Keys every /predict response carries.")
    device: str = Field(..., description="Compute device the gnn members run on.")
    load_strategy: str = Field(..., description="eager, lazy, or stream.")
