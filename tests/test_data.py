import pandas as pd
import pytest

from xauusd_neural import find_file, load_fred, load_investing, load_mt5
from xauusd_neural.synthetic import make_synthetic_dataset


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    return make_synthetic_dataset(tmp_path_factory.mktemp("data"), start="2024-01-01", end="2024-03-31")


@pytest.mark.parametrize("key", ["XAUUSD60", "XAUUSD1440", "BTCUSD1440", "us_dollar_index",
                                 "volatility_index", "wti", "US10YN", "CPI", "PPI", "NFP"])
def test_find_file_matches_every_notebook_key(dataset, key):
    assert find_file(dataset, key).exists()


def test_find_file_missing(dataset):
    with pytest.raises(FileNotFoundError):
        find_file(dataset, "does_not_exist")


def test_loaders(dataset):
    gold = load_mt5(dataset / "XAUUSD60.csv")
    assert list(gold.columns) == ["Open", "High", "Low", "Close"]
    assert gold.index.is_monotonic_increasing and gold.index.is_unique
    assert (gold["High"] >= gold[["Open", "Close"]].max(axis=1)).all()

    dxy = load_investing(find_file(dataset, "us_dollar_index"))
    assert dxy.dtype == float and dxy.index.is_monotonic_increasing

    cpi = load_fred(find_file(dataset, "CPI"), monthly=True)
    assert isinstance(cpi.index, pd.PeriodIndex)
    assert load_fred(find_file(dataset, "US10YN")).index.dtype.kind == "M"
