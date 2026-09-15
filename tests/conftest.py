"""Shared pytest fixtures for the tools.py test suite.

Uses a small, fully synthetic DataFrame instead of the downloaded dataset
so tests are fast, deterministic, and don't require network access.
"""

import pandas as pd
import pytest


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "udi": range(1, 11),
            "product_id": [f"L{100 + i}" for i in range(10)],
            "type": pd.Categorical(["L"] * 6 + ["M"] * 3 + ["H"] * 1),
            "air_temperature_k": [295.0, 296.0, 297.0, 298.0, 299.0, 300.0, 301.0, 302.0, 303.0, 304.0],
            # perfectly correlated with air_temperature_k (offset +10) for get_correlation tests
            "process_temperature_k": [305.0, 306.0, 307.0, 308.0, 309.0, 310.0, 311.0, 312.0, 313.0, 314.0],
            # one clear outlier for detect_anomalies tests
            "rotational_speed_rpm": [1500.0] * 9 + [9000.0],
            # constant column for the std == 0 edge case
            "torque_nm": [40.0] * 10,
            "tool_wear_min": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
            "machine_failure": [0] * 8 + [1, 1],
            "twf": [0] * 10,
            "hdf": [0] * 6 + [0, 0, 1, 0],
            "pwf": [0] * 9 + [1],
            "osf": [0] * 10,
            "rnf": [0] * 10,
        }
    )


@pytest.fixture(autouse=True)
def patched_dataset(monkeypatch, sample_df):
    """Point tools.py at the synthetic sample_df for every test in this suite."""
    import tools

    monkeypatch.setattr(tools, "_df", sample_df)
    yield
