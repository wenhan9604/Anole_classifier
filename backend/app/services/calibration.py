"""
Calibration utilities for classification logits/probabilities.

Supported methods:
- TemperatureScaling: single-parameter T minimizing NLL on validation set
- PlattScalingOvR: one-vs-rest logistic regression on class probabilities (binary Platt per class)
- IsotonicOvR: one-vs-rest isotonic regression on class probabilities

Persistence:
- save() / load() using JSON for TemperatureScaling, and joblib for OvR regressors.

CLI (example):
  python -m app.services.calibration_fit --method temperature \
    --logits_csv path/to/logits.csv --labels_csv path/to/labels.csv \
    --num_classes 5 --out /path/to/calibration.json
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict, Any

import json
import math


def _softmax_vec(logits: List[float]) -> List[float]:
    m = max(logits)
    exps = [math.exp(v - m) for v in logits]
    s = sum(exps)
    return [e / s for e in exps]


@dataclass
class TemperatureScaling:
    temperature: float = 1.0

    def calibrate_logits(self, logits) -> List[float]:
        if hasattr(logits, "tolist"):
            vals = logits.tolist()
            if isinstance(vals, list) and vals and isinstance(vals[0], list):
                vals = vals[0]
        else:
            vals = list(logits)
        return [v / max(self.temperature, 1e-6) for v in vals]

    def calibrate_probs(self, logits) -> List[float]:
        return _softmax_vec(self.calibrate_logits(logits))

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"method": "temperature", "temperature": float(self.temperature)}, f)

    @staticmethod
    def load(path: str) -> "TemperatureScaling":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("method") != "temperature":
            raise ValueError("Not a temperature calibration file")
        return TemperatureScaling(float(data.get("temperature", 1.0)))


class PlattScalingOvR:
    """One-vs-rest Platt scaling using logistic regression per class.

    Note: Requires scikit-learn at fit-time and load-time.
    """

    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.models = None  # set after fit

    def fit(self, logits: List[List[float]], labels: List[int]) -> None:
        from sklearn.linear_model import LogisticRegression
        import numpy as np

        X = np.array([_softmax_vec(l) for l in logits], dtype=float)
        y = np.array(labels, dtype=int)
        self.models = []
        for c in range(self.num_classes):
            # Binary labels: class c vs rest
            y_bin = (y == c).astype(int)
            # Platt on positive class probability input (use class c prob as feature)
            model = LogisticRegression(max_iter=1000)
            model.fit(X[:, c].reshape(-1, 1), y_bin)
            self.models.append(model)

    def calibrate_probs(self, logits) -> List[float]:
        if self.models is None:
            raise RuntimeError("PlattScalingOvR not fitted")
        import numpy as np
        base = _softmax_vec(logits if isinstance(logits, list) else logits.tolist())
        features = np.array(base, dtype=float)
        # Predict calibrated positives per class, renormalize
        pos = [m.predict_proba(np.array([[features[i]]]))[0, 1] for i, m in enumerate(self.models)]
        s = float(sum(pos)) or 1.0
        return [float(p) / s for p in pos]

    def save(self, path: str) -> None:
        import joblib
        joblib.dump({"method": "platt_ovr", "num_classes": self.num_classes, "models": self.models}, path)

    @staticmethod
    def load(path: str) -> "PlattScalingOvR":
        import joblib
        data = joblib.load(path)
        if data.get("method") != "platt_ovr":
            raise ValueError("Not a Platt OvR calibration file")
        calib = PlattScalingOvR(int(data["num_classes"]))
        calib.models = data["models"]
        return calib


class IsotonicOvR:
    """One-vs-rest isotonic regression per class on softmax probabilities.

    Note: Requires scikit-learn at fit-time and load-time.
    """

    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.models = None

    def fit(self, logits: List[List[float]], labels: List[int]) -> None:
        from sklearn.isotonic import IsotonicRegression
        import numpy as np

        X = np.array([_softmax_vec(l) for l in logits], dtype=float)
        y = np.array(labels, dtype=int)
        self.models = []
        for c in range(self.num_classes):
            y_bin = (y == c).astype(int)
            # Fit isotonic mapping on class-c softmax to binary labels
            ir = IsotonicRegression(out_of_bounds="clip")
            ir.fit(X[:, c], y_bin)
            self.models.append(ir)

    def calibrate_probs(self, logits) -> List[float]:
        if self.models is None:
            raise RuntimeError("IsotonicOvR not fitted")
        import numpy as np
        base = _softmax_vec(logits if isinstance(logits, list) else logits.tolist())
        cal = [float(self.models[i].predict([base[i]])[0]) for i in range(self.num_classes)]
        s = float(sum(cal)) or 1.0
        return [c / s for c in cal]

    def save(self, path: str) -> None:
        import joblib
        joblib.dump({"method": "isotonic_ovr", "num_classes": self.num_classes, "models": self.models}, path)

    @staticmethod
    def load(path: str) -> "IsotonicOvR":
        import joblib
        data = joblib.load(path)
        if data.get("method") != "isotonic_ovr":
            raise ValueError("Not an Isotonic OvR calibration file")
        calib = IsotonicOvR(int(data["num_classes"]))
        calib.models = data["models"]
        return calib


def load_calibrator(path: str):
    """Load any supported calibrator by inspecting the file.

    - JSON with {"method":"temperature"} → TemperatureScaling
    - joblib with method key → PlattScalingOvR | IsotonicOvR
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("method") == "temperature":
            return TemperatureScaling.load(path)
    except Exception:
        pass

    try:
        import joblib
        data = joblib.load(path)
        m = data.get("method")
        if m == "platt_ovr":
            return PlattScalingOvR.load(path)
        if m == "isotonic_ovr":
            return IsotonicOvR.load(path)
    except Exception as e:
        raise e

    raise ValueError("Unsupported calibration file format")


