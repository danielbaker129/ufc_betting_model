"""Isotonic calibration wrapper — importable by any script that loads a model pickle."""
import numpy as np


class IsotonicCalibrated:
    """LGBMClassifier + isotonic regressor, sklearn predict_proba-compatible."""

    def __init__(self, base, ir):
        self.base = base
        self.ir   = ir

    def predict_proba(self, X):
        raw = self.base.predict_proba(X)[:, 1]
        cal = self.ir.transform(raw)
        return np.column_stack([1 - cal, cal])
