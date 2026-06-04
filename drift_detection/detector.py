"""Adapter that wraps existing drift detector semantics."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Tuple

import numpy as np


@dataclass
class DriftDetectorAdapter:
    """Sliding-window error-rate drift detector."""

    window: int = 50
    threshold: float = 0.05
    errors: Deque[int] = field(default_factory=lambda: deque(maxlen=50))
    drift_log: List[Tuple[int, float]] = field(default_factory=list)

    def update(self, y_true: np.ndarray, y_pred: np.ndarray, chunk_id: int) -> Tuple[bool, float]:
        errors = (y_true != y_pred).astype(int)
        self.errors.extend(errors.tolist())
        if len(self.errors) == self.window:
            err_rate = float(np.mean(self.errors))
            if err_rate > self.threshold:
                self.drift_log.append((chunk_id, err_rate))
                return True, err_rate
        return False, 0.0
