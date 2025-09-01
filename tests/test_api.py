"""The service scores a trained artifact and will not invent one."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.main import create_app, load_bundle
from credit_risk.config import TrainConfig
from credit_risk.train import run
from tests.synthetic import make_transactions


def _artifact(tmp_path: Path) -> Path:
    path = tmp_path / "model.joblib"
    config = TrainConfig(
        seed=0,
        artifact_path=str(path),
        reports_dir=str(tmp_path / "reports"),
        n_clusters=2,
        min_transactions=3,
        min_level_share=0.05,
        woe_bins=3,
        lr_C=(1.0,),
        rf_max_depth=(4,),
        rf_min_samples_leaf=(1,),
        rf_n_estimators=15,
        rf_n_jobs=1,
        permutation_repeats=1,
    )
    run(config, transactions=make_transactions())
    return path


def test_missing_artifact_fails_loudly(tmp_path):
    # Arrange
    missing = tmp_path / "absent.joblib"
    # Act / Then
    with pytest.raises(FileNotFoundError, match="untrained"):
        load_bundle(missing)
    with pytest.raises(FileNotFoundError, match="untrained"):
        create_app(missing)


def test_service_source_has_no_untrained_forest():
    # The previous API constructed a RandomForestClassifier when the file
    # was missing. That class of fallback must not come back.
    source = Path(__file__).resolve().parents[1].joinpath("src", "api", "main.py").read_text()
    assert "RandomForestClassifier" not in source


def test_predict_health_and_schema(tmp_path):
    # Arrange
    app = create_app(_artifact(tmp_path))
    with TestClient(app) as client:
        # Act
        health = client.get("/health")
        schema = client.get("/features")
        columns = schema.json()["feature_columns"]
        payload = {column: 0.1 for column in columns}
        scored = client.post(
            "/predict", json={"customer_id": " CustomerId_1 ", "features": payload}
        )
        rejected = client.post(
            "/predict", json={"customer_id": "CustomerId_1", "features": {"not_a_feature": 1.0}}
        )
    # Then
    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert schema.status_code == 200
    assert scored.status_code == 200
    body = scored.json()
    assert body["customer_id"] == "CustomerId_1"
    assert 0.0 <= body["risk_probability"] <= 1.0
    assert body["is_high_risk"] in (0, 1)
    assert body["model_version"] == health.json()["model_version"]
    assert rejected.status_code == 422
    assert "missing" in rejected.json()["detail"]
