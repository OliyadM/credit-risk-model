"""Score a feature row with the trained artifact, or do not start.

An untrained estimator is not a fallback. If the artifact is missing or
incomplete the process raises during startup, before it listens. The
previous service constructed a fresh random forest in that case and
returned HTTP 200 with a sentinel score.
"""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from api.schemas import (
    FeatureSchemaResponse,
    HealthResponse,
    PredictRequest,
    PredictResponse,
)
from credit_risk.woe import WOEEncoder  # noqa: F401  (needed so joblib can unpickle)

logger = logging.getLogger("credit_risk.api")

REQUIRED_KEYS = {
    "pipeline",
    "feature_columns",
    "threshold",
    "model_name",
    "model_version",
}


def load_bundle(path: str | Path) -> dict:
    """Load the only model this service is willing to serve."""
    artifact = Path(path)
    if not artifact.is_file():
        raise FileNotFoundError(
            f"Trained model artifact is missing at {artifact}. "
            "Refusing to start with an untrained model. "
            "Run `python -m credit_risk.train --config configs/config.yaml` first."
        )
    bundle = joblib.load(artifact)
    if not isinstance(bundle, dict):
        raise ValueError(f"Artifact at {artifact} is not a model bundle")
    missing = REQUIRED_KEYS - set(bundle)
    if missing:
        raise ValueError(f"Artifact at {artifact} is missing keys: {sorted(missing)}")
    return bundle


def create_app(model_path: str | Path | None = None) -> FastAPI:
    """Build the app. Raises if the artifact cannot be loaded."""
    path = Path(
        model_path
        if model_path is not None
        else os.environ.get("MODEL_PATH", "models/credit_risk_model.joblib")
    )
    bundle = load_bundle(path)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logger.info(
        "Loaded %s %s from %s", bundle["model_name"], bundle["model_version"], path
    )
    app = FastAPI(
        title="Credit risk score",
        version=str(bundle["model_version"]),
        description=(
            "Scores a customer-level behavioral feature vector against an "
            "RFM proxy segment. The score is not a probability of default."
        ),
    )
    app.state.bundle = bundle

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            model_name=bundle["model_name"],
            model_version=bundle["model_version"],
            n_features=len(bundle["feature_columns"]),
        )

    @app.get("/features", response_model=FeatureSchemaResponse)
    def features() -> FeatureSchemaResponse:
        # The client should not have to guess the column set of this artifact.
        return FeatureSchemaResponse(
            model_name=bundle["model_name"],
            model_version=bundle["model_version"],
            threshold=float(bundle["threshold"]),
            feature_columns=list(bundle["feature_columns"]),
        )

    @app.post("/predict", response_model=PredictResponse)
    def predict(body: PredictRequest) -> PredictResponse:
        frame = _frame_from_features(body.features, bundle["feature_columns"])
        probability = float(_positive_proba(bundle["pipeline"], frame)[0])
        if not 0.0 <= probability <= 1.0:
            raise HTTPException(status_code=500, detail="Model returned a score outside [0, 1]")
        threshold = float(bundle["threshold"])
        logger.info(
            "Scored customer_id=%s probability=%.4f", body.customer_id, probability
        )
        return PredictResponse(
            customer_id=body.customer_id,
            risk_probability=probability,
            is_high_risk=int(probability >= threshold),
            threshold=threshold,
            model_name=bundle["model_name"],
            model_version=bundle["model_version"],
        )

    return app


def _frame_from_features(payload: dict[str, float], columns: list[str]) -> pd.DataFrame:
    missing = [column for column in columns if column not in payload]
    unexpected = [column for column in payload if column not in columns]
    if missing or unexpected:
        raise HTTPException(
            status_code=422,
            detail={"missing": missing, "unexpected": unexpected},
        )
    values = {}
    for column in columns:
        value = payload[column]
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise HTTPException(status_code=422, detail=f"{column} must be a finite number")
        values[column] = float(value)
    return pd.DataFrame([values], columns=columns)


def _positive_proba(pipeline, features: pd.DataFrame):
    classes = list(pipeline.named_steps["clf"].classes_)
    if 1 not in classes:
        raise HTTPException(status_code=500, detail="Loaded model has no positive class")
    return pipeline.predict_proba(features)[:, classes.index(1)]


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(create_app, factory=True, host="0.0.0.0", port=8000)
