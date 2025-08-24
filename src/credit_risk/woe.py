"""Weight of evidence with bin edges stored at fit time.

The previous encoder called qcut again inside transform. The new bins
did not match the map learned in fit, the lookup missed, and the column
became 0. Edges are part of the fitted state so a score is the same
rule on every batch.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class WOEEncoder(BaseEstimator, TransformerMixin):
    """Replace each column with its training-set weight of evidence.

    Quantile edges are taken from the fit rows only. The outer edges are
    infinite so a later value that sits outside the training range still
    lands in an end bin instead of becoming missing.
    """

    def __init__(self, n_bins: int = 5):
        self.n_bins = n_bins

    def fit(self, X, y):
        frame = self._require_frame(X)
        target = np.asarray(y).astype(int)
        if target.shape[0] != len(frame):
            raise ValueError("X and y have different lengths")
        self.feature_names_in_ = list(frame.columns)
        self.edges_ = {}
        self.woe_ = {}
        self.iv_ = {}
        for column in self.feature_names_in_:
            values = frame[column].to_numpy(dtype=float)
            edges = self._edges(values)
            bins = self._bin_index(values, edges)
            woe_map, information_value = self._bin_woe(bins, target, len(edges) - 1)
            self.edges_[column] = edges
            self.woe_[column] = woe_map
            self.iv_[column] = information_value
        return self

    def transform(self, X):
        if not hasattr(self, "edges_"):
            raise RuntimeError("WOEEncoder must be fit before transform")
        frame = self._align(X)
        encoded = {}
        for column in self.feature_names_in_:
            values = frame[column].to_numpy(dtype=float)
            bins = self._bin_index(values, self.edges_[column])
            mapping = self.woe_[column]
            encoded[column] = [mapping.get(int(bin_id), 0.0) for bin_id in bins]
        return pd.DataFrame(encoded, index=frame.index)

    def _edges(self, values: np.ndarray) -> np.ndarray:
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return np.array([-np.inf, np.inf])
        quantiles = np.linspace(0.0, 1.0, self.n_bins + 1)[1:-1]
        inner = np.unique(np.quantile(finite, quantiles))
        inner = inner[np.isfinite(inner)]
        return np.concatenate([[-np.inf], inner, [np.inf]])

    def _bin_index(self, values: np.ndarray, edges: np.ndarray) -> np.ndarray:
        clean = np.where(np.isfinite(values), values, 0.0)
        inner = edges[1:-1]
        if inner.size == 0:
            return np.zeros(clean.shape[0], dtype=int)
        return np.digitize(clean, inner, right=False).astype(int)

    def _bin_woe(
        self, bins: np.ndarray, target: np.ndarray, n_bins: int
    ) -> tuple[dict[int, float], float]:
        """Smoothed WOE. Empty bins stay finite instead of becoming +/- inf.

        A bin with no events would otherwise dominate the linear model.
        The 0.5 is Laplace smoothing, not a claim about the true rate.
        """
        n_good = max(int(np.sum(target == 0)), 1)
        n_bad = max(int(np.sum(target == 1)), 1)
        woe_map: dict[int, float] = {}
        information_value = 0.0
        for bin_id in range(n_bins):
            mask = bins == bin_id
            good = float(np.sum(mask & (target == 0)))
            bad = float(np.sum(mask & (target == 1)))
            dist_good = (good + 0.5) / (n_good + 0.5 * n_bins)
            dist_bad = (bad + 0.5) / (n_bad + 0.5 * n_bins)
            woe = float(np.log(dist_good / dist_bad))
            woe_map[bin_id] = woe
            information_value += (dist_good - dist_bad) * woe
        return woe_map, float(information_value)

    def _require_frame(self, X) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("WOEEncoder.fit expects a pandas DataFrame with named columns")
        return X.reset_index(drop=True)

    def _align(self, X) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            missing = [column for column in self.feature_names_in_ if column not in X.columns]
            if missing:
                raise ValueError(f"Missing features for WOE: {missing}")
            return X.loc[:, self.feature_names_in_]
        array = np.asarray(X, dtype=float)
        if array.ndim != 2 or array.shape[1] != len(self.feature_names_in_):
            raise ValueError(
                f"Expected {len(self.feature_names_in_)} columns, got shape {array.shape}"
            )
        return pd.DataFrame(array, columns=self.feature_names_in_)
