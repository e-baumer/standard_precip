from pathlib import Path

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@pytest.fixture
def monthly_df():
    return pd.read_csv(DATA_DIR / "monthly_data.csv")


@pytest.fixture
def wichita_df():
    return pd.read_csv(DATA_DIR / "wichita_rain.csv")


@pytest.fixture
def daily_test_df():
    return pd.read_csv(DATA_DIR / "daily_data_test.csv")
