import pandas as pd
import pytest

from app.services.anomaly_service import _add_engineered_features


def test_engineered_features_expose_spike_weekend_and_rare_pair():
    rows = [
        {"amount": 100.0, "day_of_week": 1, "department": "IT", "category": "Hardware"}
        for _ in range(10)
    ]
    rows.append(  # index 10: a spike on a Saturday
        {"amount": 900.0, "day_of_week": 5, "department": "IT", "category": "Hardware"}
    )
    rows.append(  # index 11: a pair that appears only once
        {"amount": 100.0, "day_of_week": 2, "department": "HR", "category": "Cloud"}
    )

    features = _add_engineered_features(pd.DataFrame(rows))

    assert features.iloc[10]["amount_ratio"] == pytest.approx(9.0)
    assert features.iloc[10]["is_weekend"] == 1
    assert features.iloc[0]["is_weekend"] == 0
    assert features.iloc[11]["pair_count"] == 1
    assert features.iloc[0]["pair_count"] == 11