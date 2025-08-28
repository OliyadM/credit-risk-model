"""A small seeded run writes a report and an artifact, and does not leak RFM."""

from credit_risk.config import TrainConfig
from credit_risk.evaluate import CAVEAT
from credit_risk.features import assert_features_allowed
from credit_risk.train import run
from tests.synthetic import make_transactions


def test_smoke_train_writes_report_and_refuses_label_inputs(tmp_path):
    # Arrange
    config = TrainConfig(
        seed=0,
        artifact_path=str(tmp_path / "model.joblib"),
        reports_dir=str(tmp_path / "reports"),
        n_clusters=2,
        min_transactions=3,
        min_level_share=0.05,
        woe_bins=3,
        lr_C=(1.0,),
        rf_max_depth=(4,),
        rf_min_samples_leaf=(1,),
        rf_n_estimators=20,
        rf_n_jobs=1,
        permutation_repeats=2,
    )
    # Act
    metrics = run(config, transactions=make_transactions())
    # Then
    assert (tmp_path / "reports" / "metrics.json").is_file()
    assert (tmp_path / "reports" / "model_card.md").is_file()
    assert (tmp_path / "reports" / "figures" / "roc_curve.png").is_file()
    assert metrics["caveat"] == CAVEAT
    assert "not a probability of default" in (tmp_path / "reports" / "model_card.md").read_text()
    assert metrics["test"]["baseline"]["roc_auc"] == 0.5
    assert 0.0 <= metrics["test"][metrics["champion"]]["roc_auc"] <= 1.0
    assert "rfm_oracle_diagnostic" in metrics["test"]
    assert_features_allowed([row["feature"] for row in metrics["permutation_importance"]])
    assert (tmp_path / "model.joblib").is_file()
