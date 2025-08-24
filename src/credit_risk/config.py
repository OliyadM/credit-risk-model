"""Load the training config.

Paths, the split, the grid, and the cost weights live in one file so a
rerun is the same run. Code defaults exist so tests can build a small
config without writing YAML.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import yaml


@dataclass
class TrainConfig:
    seed: int = 42
    raw_path: str = "data/raw/data.csv"
    artifact_path: str = "models/credit_risk_model.joblib"
    reports_dir: str = "reports"
    val_fraction: float = 0.2
    test_fraction: float = 0.2
    min_transactions: int = 3
    n_clusters: int = 3
    min_level_share: float = 0.01
    timezone_offset_hours: int = 3
    woe_bins: int = 5
    lr_C: tuple[float, ...] = (0.1, 1.0, 10.0)
    rf_max_depth: tuple[int, ...] = (6, 12)
    rf_min_samples_leaf: tuple[int, ...] = (5, 20)
    rf_n_estimators: int = 200
    rf_n_jobs: int = -1
    prefer_logistic_margin: float = 0.01
    fp_cost: float = 1.0
    fn_cost: float = 5.0
    permutation_repeats: int = 5


_SECTIONS = {
    "seed": {"seed"},
    "paths": {"raw", "artifact", "reports"},
    "split": {"val_fraction", "test_fraction"},
    "target": {"n_clusters", "min_transactions"},
    "features": {"min_level_share", "timezone_offset_hours"},
    "models": {
        "woe_bins",
        "lr_C",
        "rf_max_depth",
        "rf_min_samples_leaf",
        "rf_n_estimators",
        "rf_n_jobs",
        "prefer_logistic_margin",
    },
    "cost": {"false_positive", "false_negative"},
    "evaluation": {"permutation_repeats"},
}


def load_config(path: str | Path) -> TrainConfig:
    """Read YAML into a TrainConfig. Unknown keys raise, so a typo fails the run."""
    config_path = Path(path)
    with config_path.open() as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"Config at {config_path} must be a mapping")
    unknown = set(raw) - set(_SECTIONS)
    if unknown:
        raise ValueError(f"Unknown config sections: {sorted(unknown)}")
    for section, allowed in _SECTIONS.items():
        block = raw.get(section, {})
        if block is None:
            block = {}
        extra = set(block) - allowed if section != "seed" else set()
        if section == "seed":
            continue
        if extra:
            raise ValueError(f"Unknown keys in {section}: {sorted(extra)}")

    paths = raw.get("paths") or {}
    split = raw.get("split") or {}
    target = raw.get("target") or {}
    features = raw.get("features") or {}
    models = raw.get("models") or {}
    cost = raw.get("cost") or {}
    evaluation = raw.get("evaluation") or {}
    config = TrainConfig(
        seed=int(raw.get("seed", 42)),
        raw_path=str(paths.get("raw", "data/raw/data.csv")),
        artifact_path=str(paths.get("artifact", "models/credit_risk_model.joblib")),
        reports_dir=str(paths.get("reports", "reports")),
        val_fraction=float(split.get("val_fraction", 0.2)),
        test_fraction=float(split.get("test_fraction", 0.2)),
        min_transactions=int(target.get("min_transactions", 3)),
        n_clusters=int(target.get("n_clusters", 3)),
        min_level_share=float(features.get("min_level_share", 0.01)),
        timezone_offset_hours=int(features.get("timezone_offset_hours", 3)),
        woe_bins=int(models.get("woe_bins", 5)),
        lr_C=tuple(float(value) for value in models.get("lr_C", [0.1, 1.0, 10.0])),
        rf_max_depth=tuple(int(value) for value in models.get("rf_max_depth", [6, 12])),
        rf_min_samples_leaf=tuple(
            int(value) for value in models.get("rf_min_samples_leaf", [5, 20])
        ),
        rf_n_estimators=int(models.get("rf_n_estimators", 200)),
        rf_n_jobs=int(models.get("rf_n_jobs", -1)),
        prefer_logistic_margin=float(models.get("prefer_logistic_margin", 0.01)),
        fp_cost=float(cost.get("false_positive", 1.0)),
        fn_cost=float(cost.get("false_negative", 5.0)),
        permutation_repeats=int(evaluation.get("permutation_repeats", 5)),
    )
    _validate(config)
    return config


def _validate(config: TrainConfig) -> None:
    names = {item.name for item in fields(config)}
    if set(names) != {item.name for item in fields(TrainConfig)}:
        raise ValueError("Config is missing fields")
    if not 0 < config.val_fraction < 1 or not 0 < config.test_fraction < 1:
        raise ValueError("Split fractions must be between 0 and 1")
    if config.val_fraction + config.test_fraction >= 1:
        raise ValueError("val_fraction + test_fraction leaves no fit cohort")
    if config.min_transactions < 2:
        raise ValueError("min_transactions must be at least 2")
    if config.n_clusters < 2:
        raise ValueError("n_clusters must be at least 2")
    if config.fp_cost < 0 or config.fn_cost < 0:
        raise ValueError("Costs cannot be negative")
