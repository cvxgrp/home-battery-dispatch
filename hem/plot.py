"""Plotting utilities."""

from math import sqrt
import calendar

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def latexify(
    fig_width: float | None = None,
    fig_height: float | None = None,
    font_size: int = 10,
) -> None:
    """Configure matplotlib for LaTeX-style plots."""
    if fig_width is None:
        fig_width = 5.0

    if fig_height is None:
        golden_mean = (sqrt(5) - 1.0) / 2.0
        fig_height = fig_width * golden_mean

    params = {
        "backend": "ps",
        "text.latex.preamble": r"\usepackage{gensymb} \usepackage{amsmath}",
        "axes.labelsize": font_size,
        "axes.titlesize": font_size,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
        "legend.fontsize": font_size,
        "figure.figsize": [fig_width, fig_height],
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": "Computer Modern Roman",
        "savefig.dpi": 200,
        "savefig.format": "pdf",
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.2,
    }
    plt.rcParams.update(params)


class JanYearDateFormatter(mdates.DateFormatter):
    """Date formatter that shows year only for January."""

    def __init__(self, fmt: str = "%b"):
        super().__init__(fmt)

    def __call__(self, x, pos=0):
        dt = mdates.num2date(x)
        if dt.month == 1:
            return dt.strftime("%b\n%Y")
        return dt.strftime("%b")


def plot_multiyear(
    data: pd.Series,
    ylabel: str,
    save_path: str | None = None,
) -> None:
    """Plot time series spanning multiple years."""
    fig, ax = plt.subplots()
    ax.plot(data, color="#0072B2")
    ax.set_ylabel(ylabel)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(JanYearDateFormatter())

    if save_path:
        plt.savefig(save_path)
    plt.close()


def plot_one_week(
    data: pd.Series,
    start_date: str,
    end_date: str,
    ylabel: str,
    save_path: str | None = None,
) -> None:
    """Plot one week of data."""
    week = data[start_date:end_date]

    fig, ax = plt.subplots()
    ax.plot(week.values, color="#0072B2")
    ax.set_ylabel(ylabel)
    ax.set_xticks(np.arange(0, 24 * 7 + 1, 24))
    ax.set_xticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun", "Mon"])

    if save_path:
        plt.savefig(save_path)
    plt.close()


def plot_load_year(
    load: np.ndarray,
    index: pd.DatetimeIndex,
    save_path: str | None = None,
) -> None:
    """Plot load over a year."""
    fig, ax = plt.subplots()

    ax.plot(index, load, color="#0072B2")

    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.set_ylabel("Load (kW)")

    if save_path:
        plt.savefig(save_path)
    plt.close()


def plot_grid_power(
    p_storage: np.ndarray,
    index: pd.DatetimeIndex,
    tier_thresholds: list[float] | None = None,
    save_path: str | None = None,
) -> None:
    """Plot grid power for a single policy over a year."""
    fig, ax = plt.subplots()

    ax.plot(index, p_storage, color="#0072B2")

    if tier_thresholds:
        for thresh in tier_thresholds[:3]:
            ax.axhline(thresh, color="black", linestyle="dashed", lw=0.8)

    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.set_ylabel("Grid power (kW)")

    if save_path:
        plt.savefig(save_path)
    plt.close()


def plot_charge_level_year(
    q_storage: np.ndarray,
    index: pd.DatetimeIndex,
    capacity: float,
    save_path: str | None = None,
) -> None:
    """Plot charge level over a year."""
    fig, ax = plt.subplots()

    ax.plot(index, q_storage[:-1], color="#0072B2")
    ax.set_ylim(0, capacity)

    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.set_ylabel("Charge level (kWh)")

    if save_path:
        plt.savefig(save_path)
    plt.close()


def _add_price_overlay(ax, prices: np.ndarray) -> None:
    """Add price overlay on secondary y-axis."""
    ax2 = ax.twinx()
    ax2.plot(prices, color="k", linestyle="dotted")
    ax2.set_ylabel("Price (NOK/kWh)")
    ax2.grid(False)


def _plot_week(
    values: np.ndarray,
    prices: np.ndarray,
    ylabel: str,
    save_path: str | None = None,
    ylim: tuple[float, float] | None = None,
) -> None:
    """Plot a weekly series with price overlay."""
    fig, ax = plt.subplots()
    ax.plot(values, color="#0072B2")
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(ylim)
    ax.set_xticks(np.arange(0, 24 * 7 + 1, 24))
    ax.set_xticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun", "Mon"])
    _add_price_overlay(ax, prices)
    if save_path:
        plt.savefig(save_path)
    plt.close()


def plot_week_load(
    load: np.ndarray,
    prices: np.ndarray,
    save_path: str | None = None,
    ylim: tuple[float, float] | None = None,
) -> None:
    """Plot load for one week with price overlay."""
    _plot_week(load, prices, "Load (kW)", save_path, ylim)


def plot_week_grid_power(
    grid_power: np.ndarray,
    prices: np.ndarray,
    save_path: str | None = None,
    ylim: tuple[float, float] | None = None,
) -> None:
    """Plot grid power for one week with price overlay."""
    _plot_week(grid_power, prices, "Grid power (kW)", save_path, ylim)


def plot_week_soc(
    q_storage: np.ndarray,
    prices: np.ndarray,
    save_path: str | None = None,
) -> None:
    """Plot charge level for one week with price overlay."""
    _plot_week(q_storage, prices, "Charge level (kWh)", save_path)


def plot_z_comparison(
    z_no_storage: list[float],
    z_storage: list[float],
    tier_thresholds: list[float] | None = None,
    labels: tuple[str, str] = ("No storage", "Prescient"),
    save_path: str | None = None,
) -> None:
    """Plot monthly z values comparison."""
    months = np.arange(1, 13)
    bar_width = 0.35

    fig, ax = plt.subplots()
    ax.bar(
        months - bar_width / 2,
        z_no_storage,
        bar_width,
        color="#D55E00",
        label=labels[0],
    )
    ax.bar(
        months + bar_width / 2, z_storage, bar_width, color="#0072B2", label=labels[1]
    )

    if tier_thresholds:
        for thresh in tier_thresholds[:3]:
            ax.axhline(thresh, color="black", linestyle="dashed", lw=0.8)

    ax.set_xticks(months)
    ax.set_xticklabels([calendar.month_abbr[m] for m in months])
    ax.set_ylabel("Peak power average (kW)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.95))

    if save_path:
        plt.savefig(save_path)
    plt.close()


def plot_cost_vs_capacity(
    capacities: np.ndarray,
    savings_dict: dict[str, np.ndarray],
    save_path: str | None = None,
) -> None:
    """Plot annual savings vs storage capacity."""
    fig, ax = plt.subplots()

    colors = {"2020": "#009E73", "2021": "#D55E00", "2022": "#0072B2"}

    for year, savings_pct in savings_dict.items():
        ax.plot(capacities, savings_pct, color=colors.get(year, "#0072B2"), label=year)

    ax.set_xlabel("Storage capacity (kWh)")
    ax.set_ylabel(r"Annual savings (\%)")
    ax.legend()

    if save_path:
        plt.savefig(save_path)
    plt.close()


def plot_baseline_comparison(
    actual: pd.Series,
    baseline: pd.Series,
    start_date: str,
    end_date: str,
    ylabel: str,
    save_path: str | None = None,
) -> None:
    """Plot actual vs baseline for one week."""
    golden_mean = (sqrt(5) - 1.0) / 2.0

    fig, ax = plt.subplots(figsize=(5, 5 * golden_mean / 2))
    ax.plot(actual[start_date:end_date].values, color="#0072B2", label="Actual")
    ax.plot(baseline[start_date:end_date].values, color="#D55E00", label="Baseline")
    ax.legend(
        ncol=2,
        loc="lower center",
        bbox_to_anchor=(0.51, 0.9),
        bbox_transform=plt.gcf().transFigure,
    )
    ax.set_ylabel(ylabel)
    ax.set_xticks(np.arange(0, 24 * 7 + 1, 24))
    ax.set_xticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun", "Mon"])

    if save_path:
        plt.savefig(save_path)
    plt.close()


def plot_forecast_comparison(
    past: np.ndarray,
    future: np.ndarray,
    baseline_forecast: np.ndarray,
    ar_forecast: np.ndarray,
    past_index: pd.DatetimeIndex,
    future_index: pd.DatetimeIndex,
    ylabel: str,
    save_path: str | None = None,
) -> None:
    """Plot baseline vs baseline+AR forecast."""
    golden_mean = (sqrt(5) - 1.0) / 2.0

    fig, ax = plt.subplots(figsize=(5, 5 * golden_mean / 1.2))

    ax.plot(past_index, past, color="k", label="Past")
    ax.axvline(past_index[-1], color="gray")
    ax.plot(future_index, future, color="k", linestyle="dashed", label="Future")
    ax.plot(future_index, baseline_forecast, color="#0072B2", label="Baseline")
    ax.plot(future_index, ar_forecast, color="#D55E00", label="Forecast")

    ax.set_ylabel(ylabel)
    ax.legend(bbox_to_anchor=(0.5, 1.01), loc="lower center", ncol=2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

    if save_path:
        plt.savefig(save_path)
    plt.close()
