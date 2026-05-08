"""Опційний скрипт для генерації HTML-звіту Evidently.

Перед запуском встановіть додаткові залежності:
    pip install -r requirements-evidently.txt
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    """Генерує drift_report.html для штучно зміщеної live-вибірки."""
    ref_data = joblib.load(ROOT / "reference_stats.joblib")
    ref_df = pd.DataFrame(ref_data["X"], columns=ref_data["feature_names"])

    current = ref_df.copy().sample(n=min(100, len(ref_df)), random_state=0, replace=True)
    current["petal_length"] = current["petal_length"] + 1.5

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref_df, current_data=current)

    output = ROOT / "drift_report.html"
    report.save_html(str(output))
    print(f"Report saved to {output}")


if __name__ == "__main__":
    main()
