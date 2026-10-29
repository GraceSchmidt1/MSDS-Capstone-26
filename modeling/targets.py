"""
High- vs. low-engagement labels for the Stage 1 classifier.

A label rule is learned from training posts only and then applied to any posts, so held-out posts never
influence their own labels:

    rule = make_label_rule("reach_adjusted", config).fit(fs, train_idx)
    y_train, y_test = rule.label(fs, train_idx), rule.label(fs, test_idx)

Rules are fit separately per platform (in the pooled model each platform gets its own threshold / reach
line), so "high" always means high *for that platform*.

  quantile        label = 1 if rate > the q-quantile of training rates (q = 0.5: above median;
                  q = 0.6667: top third). Posts exactly at the threshold are 0.
  reach_adjusted  fit the median rate as a + b·log2(impressions) on training posts (exact quantile
                  regression, robust to extreme small-reach posts); label = 1 if the post's residual is above
                  the median training residual, i.e. it beat the typical rate for its reach. Using the median
                  residual as the cut keeps the classes ~50/50. Posts at the cut (within 1e-9) are 0.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from modeling.features import FeatureSet


def row_platforms(fs: FeatureSet, config: dict) -> np.ndarray:
    """Platform of each row: the FeatureSet's platform, or decoded from the pooled platform dummies."""
    if fs.platform in config["platforms"]:
        return np.full(len(fs.y), fs.platform, dtype=object)
    out = np.full(len(fs.y), config["platforms"][0], dtype=object)    # reference platform has no dummy
    for p in config["platforms"][1:]:
        out[fs.X[f"Platform: {p}"].to_numpy() == 1] = p
    return out


class LabelRule:
    """Base class: per-platform parameters learned in fit(), applied in label()."""

    name = "label"

    def __init__(self, config: dict):
        self.config = config
        self.params: dict[str, dict] = {}

    def fit(self, fs: FeatureSet, idx=None) -> "LabelRule":
        idx = np.arange(len(fs.y)) if idx is None else np.asarray(idx)
        plats = row_platforms(fs, self.config)
        y = fs.y.to_numpy()
        reach = fs.X["log2_impressions"].to_numpy() if "log2_impressions" in fs.X else None
        self.params = {}
        for p in np.unique(plats[idx]):
            rows = idx[plats[idx] == p]
            self.params[p] = self._fit_platform(y[rows], None if reach is None else reach[rows])
        return self

    def label(self, fs: FeatureSet, idx=None) -> np.ndarray:
        if not self.params:
            raise RuntimeError("fit() the rule before labelling")
        idx = np.arange(len(fs.y)) if idx is None else np.asarray(idx)
        plats = row_platforms(fs, self.config)[idx]
        y = fs.y.to_numpy()[idx]
        reach = fs.X["log2_impressions"].to_numpy()[idx] if "log2_impressions" in fs.X else None
        out = np.zeros(len(idx), dtype=int)
        for p in np.unique(plats):
            if p not in self.params:
                raise ValueError(f"No label rule fitted for platform {p}")
            m = plats == p
            out[m] = self._label_platform(self.params[p], y[m], None if reach is None else reach[m])
        return out

    def describe(self) -> pd.DataFrame:
        return pd.DataFrame([{"rule": self.name, "platform": p, **v} for p, v in self.params.items()])


class QuantileRule(LabelRule):
    def __init__(self, config: dict, q: float, name: str):
        super().__init__(config)
        self.q, self.name = q, name

    def _fit_platform(self, y, reach):
        return {"threshold_pct": float(np.quantile(y, self.q))}

    def _label_platform(self, params, y, reach):
        return (y > params["threshold_pct"]).astype(int)


class ReachAdjustedRule(LabelRule):
    def __init__(self, config: dict, name: str):
        super().__init__(config)
        self.name = name

    # A median line passes exactly through some training posts; residuals within this of the cut count as
    # ties (label 0, as for quantile rules) so float noise never decides a label.
    TIE_TOL = 1e-9

    def _fit_platform(self, y, reach):
        if reach is None:
            raise ValueError("reach_adjusted labels need log2_impressions in the features")
        # Median (quantile) regression, not least squares: a few small-reach posts with extreme rates
        # (e.g. 3 impressions → 66.7%) would otherwise tilt the line and make the label track reach.
        # Solved exactly as a linear program (HiGHS), so there is no iteration limit and reruns agree.
        from sklearn.linear_model import QuantileRegressor
        fit = QuantileRegressor(quantile=0.5, alpha=0, solver="highs").fit(reach.reshape(-1, 1), y)
        intercept, slope = float(fit.intercept_), float(fit.coef_[0])
        resid = y - (intercept + slope * reach)
        return {"intercept": intercept, "slope_per_doubling": slope,
                "residual_threshold": float(np.median(resid))}

    def _label_platform(self, params, y, reach):
        expected = params["intercept"] + params["slope_per_doubling"] * reach
        return ((y - expected) - params["residual_threshold"] > self.TIE_TOL).astype(int)


def make_label_rule(name: str, config: dict) -> LabelRule:
    spec = config["classification"]["labels"][name]
    if spec["type"] == "quantile":
        return QuantileRule(config, float(spec["q"]), name)
    if spec["type"] == "reach_adjusted":
        return ReachAdjustedRule(config, name)
    raise ValueError(f"Unknown label type: {spec['type']}")


def label_summary(fs: FeatureSet, config: dict) -> pd.DataFrame:
    """Full-data label counts and share labelled high, per label definition and platform (for reporting)."""
    rows = []
    plats = row_platforms(fs, config)
    for name in config["classification"]["labels"]:
        lab = make_label_rule(name, config).fit(fs).label(fs)
        for p in np.unique(plats):
            m = plats == p
            rows.append({"label": name, "platform": p, "posts": int(m.sum()), "high": int(lab[m].sum()),
                         "share_high": float(lab[m].mean())})
    return pd.DataFrame(rows)
