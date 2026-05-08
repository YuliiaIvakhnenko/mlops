"""Детекція data drift на основі KS-тесту."""

from typing import Dict, List

import numpy as np
from scipy import stats


class DriftDetector:
    """Простий статистичний детектор drift для числових ознак."""

    def __init__(self, reference: np.ndarray, feature_names: List[str]) -> None:
        """Ініціалізує детектор reference-вибіркою.

        Args:
            reference: Двовимірний масив навчальних даних.
            feature_names: Назви ознак.

        Raises:
            ValueError: Якщо форма reference не відповідає очікуваній.
        """
        if reference.ndim != 2:
            raise ValueError("reference must be 2D (n_samples, n_features)")
        if reference.shape[1] != len(feature_names):
            raise ValueError("feature_names length must match reference columns")

        self.reference = reference
        self.feature_names = feature_names

    def detect(self, current: np.ndarray, alpha: float = 0.05) -> dict:
        """Перевіряє drift для кожної ознаки через KS-тест.

        Args:
            current: Поточна live-вибірка.
            alpha: Поріг значущості.

        Returns:
            Словник із загальним прапорцем drift та деталями по кожній ознаці.
        """
        if current.ndim != 2 or current.shape[1] != self.reference.shape[1]:
            raise ValueError(
                f"current must be 2D with {self.reference.shape[1]} columns"
            )

        per_feature: Dict[str, dict] = {}
        drifted: List[str] = []

        for index, name in enumerate(self.feature_names):
            ref_col = self.reference[:, index]
            cur_col = current[:, index]

            ks_stat, p_value = stats.ks_2samp(ref_col, cur_col)
            is_drift = bool(p_value < alpha)

            per_feature[name] = {
                "statistic": float(ks_stat),
                "p_value": float(p_value),
                "drift_detected": is_drift,
            }

            if is_drift:
                drifted.append(name)

        return {
            "drift_detected": len(drifted) > 0,
            "n_drifted_features": len(drifted),
            "drifted_features": drifted,
            "per_feature": per_feature,
            "n_samples": int(current.shape[0]),
            "alpha": alpha,
        }
