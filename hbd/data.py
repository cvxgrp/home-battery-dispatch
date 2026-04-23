"""Data loading utilities."""

from pathlib import Path

import pandas as pd


def load_data(data_dir: Path | str = "data") -> dict:
    """Load all time series data.

    Args:
        data_dir: Path to data directory.

    Returns:
        Dictionary with load, tou_prices, da_prices series.
    """
    data_dir = Path(data_dir)

    load = pd.read_csv(data_dir / "loads.csv", parse_dates=[0], index_col=0)[
        "Load (kW)"
    ]

    tou_prices = pd.read_csv(data_dir / "tou_prices.csv", parse_dates=[0], index_col=0)[
        "TOU Price (NOK/kWh)"
    ]

    da_prices = pd.read_csv(data_dir / "da_prices.csv", parse_dates=[0], index_col=0)[
        "DA Price (NOK/kWh)"
    ]

    return {"load": load, "tou_prices": tou_prices, "da_prices": da_prices}


def load_forecaster(data_dir: Path | str = "data") -> dict:
    """Load pre-trained forecaster parameters.

    Args:
        data_dir: Path to data directory.

    Returns:
        Dictionary with baselines and AR parameters.
    """
    data_dir = Path(data_dir)

    load_baseline = pd.read_csv(
        data_dir / "load_baseline.csv", parse_dates=[0], index_col=0
    )["Baseline load (kW)"]

    da_price_baseline = pd.read_csv(
        data_dir / "da_price_baseline.csv", parse_dates=[0], index_col=0
    )["Baseline day-ahead price (NOK/kWh)"]

    load_ar_params = pd.read_pickle(data_dir / "load_AR_params.pkl")
    da_price_ar_params = pd.read_pickle(data_dir / "da_price_AR_params.pkl")

    return {
        "load_baseline": load_baseline,
        "da_price_baseline": da_price_baseline,
        "load_ar_params": load_ar_params,
        "da_price_ar_params": da_price_ar_params,
    }


def get_eval_window(data: dict, year: int = 2022) -> dict:
    """Extract evaluation window for a given year.

    Args:
        data: Dictionary from load_data().
        year: Year to extract.

    Returns:
        Dictionary with arrays and datetime index for evaluation.
    """
    load = data["load"]
    tou_prices = data["tou_prices"]
    da_prices = data["da_prices"]

    start = pd.Timestamp(f"{year}-01-01 00:00:00")
    T = 24 * 365

    start_idx = load.index.get_loc(start)
    end_idx = start_idx + T

    datetime_index = load.index[start_idx:end_idx]

    return {
        "load": load.iloc[start_idx:end_idx].values,
        "tou_prices": tou_prices.iloc[start_idx:end_idx].values,
        "da_prices": da_prices.iloc[start_idx:end_idx].values,
        "datetime_index": datetime_index,
        "start_idx": start_idx,
        "T": T,
    }
