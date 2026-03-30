"""Policies."""

import numpy as np
import pandas as pd
import cvxpy as cp
from tqdm import trange

from hem.forecast import make_load_forecast, make_price_forecast

# Storage parameters
CAPACITY = 40  # Storage capacity (kWh)
MAX_CHARGE = 20  # Max charge rate (kW)
MAX_DISCHARGE = 20  # Max discharge rate (kW)
MAX_GRID_POWER = 20  # Max grid power (kW)

# Efficiency parameters
ETA_STORE = 0.99998  # Storage efficiency (per hour)
ETA_CHARGE = 0.95  # Charging efficiency
ETA_DISCHARGE = 0.95  # Discharging efficiency

# Peak power tariff tiers (Norway)
TIER_COSTS = [83, 147, 252, 371, 490]  # NOK per month
TIER_THRESHOLDS = [2, 5, 10, 15, 20]  # kW

# MPC parameters
HORIZON = 24 * 30  # Planning horizon (hours)


def _enforce_grid_limits(load_t: float, c_t: float, d_t: float):
    """Clamp grid power to [0, MAX_GRID_POWER], adjusting c/d as needed."""
    p_t = load_t + c_t - d_t
    if p_t > MAX_GRID_POWER:
        c_t = max(0, MAX_GRID_POWER - load_t + d_t)
        p_t = load_t + c_t - d_t
    if p_t < 0:
        d_t = load_t + c_t
        p_t = 0
    return p_t, c_t, d_t


def energy_cost(
    p: cp.Variable, tou_prices: np.ndarray, da_prices: np.ndarray
) -> cp.Expression:
    """Energy cost: sum of TOU and DA prices times power."""
    return cp.sum(cp.multiply(tou_prices + da_prices, p))


def peak_power_cost(
    p: cp.Variable,
    datetime_index: pd.DatetimeIndex,
    N: int = 3,
    p_prev: list | None = None,
    datetime_index_prev: pd.DatetimeIndex | None = None,
) -> tuple[cp.Expression, list, list[cp.Expression]]:
    """Peak power cost with tiered pricing.

    Args:
        p: Power variable.
        datetime_index: Datetime index for planning horizon.
        N: Number of largest daily peaks to average.
        p_prev: Previous power values in current month (for MPC).
        datetime_index_prev: Datetime index for previous values.

    Returns:
        Tuple of (cost_expression, constraints, z_values).
    """
    tier_costs = np.array(TIER_COSTS)
    tier_thresholds = np.array(TIER_THRESHOLDS)

    # Extend with previous values if provided
    if p_prev and datetime_index_prev is not None:
        datetime_index_full = datetime_index_prev.append(datetime_index)
        p_full = cp.hstack([p_prev, p])
    else:
        datetime_index_full = datetime_index
        p_full = p

    cost = 0
    constraints = []
    z_values = []

    for month in pd.unique(datetime_index_full.month):
        month_mask = datetime_index_full.month == month
        days = pd.unique(datetime_index_full[month_mask].date)

        daily_max = [
            cp.max(p_full[(datetime_index_full.date == day) & month_mask])
            for day in days
        ]
        n_days = len(days)
        z = cp.sum_largest(cp.hstack(daily_max), min(N, n_days)) / min(N, n_days)
        z_values.append(z)

        # Binary variable for tier selection
        s = cp.Variable(len(tier_costs), boolean=True)
        cost += tier_costs @ s
        constraints += [z <= tier_thresholds @ s, cp.sum(s) == 1]

    return cost, constraints, z_values


def compute_peak_power_cost_value(z: float) -> float:
    """Compute peak power cost for a realized z value."""
    for threshold, cost in zip(TIER_THRESHOLDS, TIER_COSTS):
        if z <= threshold:
            return cost
    return TIER_COSTS[-1]


def get_z_values(
    power: np.ndarray,
    datetime_index: pd.DatetimeIndex,
    N: int = 3,
) -> list[float]:
    """Compute monthly z values from realized power trajectory."""
    z_values = []

    for month in pd.unique(datetime_index.month):
        month_mask = datetime_index.month == month
        unique_days = pd.unique(datetime_index[month_mask].date)

        daily_peaks = []
        for day in unique_days:
            day_mask = datetime_index.date == day
            daily_peak = power[month_mask & day_mask].max()
            daily_peaks.append(daily_peak)

        n_largest = sorted(daily_peaks, reverse=True)[:N]
        z = sum(n_largest) / min(N, len(daily_peaks))
        z_values.append(z)

    return z_values


def compute_costs(
    tou_prices: np.ndarray,
    da_prices: np.ndarray,
    power: np.ndarray,
    datetime_index: pd.DatetimeIndex,
    N: int = 3,
) -> dict:
    """Compute all cost components from realized power trajectory."""
    monthly_tou = []
    monthly_da = []
    monthly_peak = []

    z_values = get_z_values(power, datetime_index, N)

    for i, month in enumerate(pd.unique(datetime_index.month)):
        mask = datetime_index.month == month
        monthly_tou.append(np.sum(tou_prices[mask] * power[mask]))
        monthly_da.append(np.sum(da_prices[mask] * power[mask]))
        monthly_peak.append(compute_peak_power_cost_value(z_values[i]))

    return {
        "monthly_tou": monthly_tou,
        "monthly_da": monthly_da,
        "monthly_peak": monthly_peak,
        "tou": sum(monthly_tou),
        "da": sum(monthly_da),
        "peak": sum(monthly_peak),
        "total": sum(monthly_tou) + sum(monthly_da) + sum(monthly_peak),
    }


def optimize(
    load: np.ndarray,
    tou_prices: np.ndarray,
    da_prices: np.ndarray,
    datetime_index: pd.DatetimeIndex,
    capacity: float = CAPACITY,
    max_charge: float = MAX_CHARGE,
    max_discharge: float = MAX_DISCHARGE,
    q_init: float | None = None,
    q_final: float | None = None,
    include_peak_power: bool = True,
    peak_power_N: int = 3,
    p_prev: list | None = None,
    datetime_index_prev: pd.DatetimeIndex | None = None,
    solver=None,
    verbose: bool = False,
) -> dict:
    """Solve energy scheduling optimization.

    Args:
        load: Load array.
        tou_prices: TOU prices array.
        da_prices: DA prices array.
        datetime_index: Datetime index.
        capacity: Storage capacity (kWh).
        max_charge: Max charge rate (kW).
        max_discharge: Max discharge rate (kW).
        q_init: Initial charge level (default: capacity/2).
        q_final: Final charge level (default: capacity/2).
        include_peak_power: Whether to include peak power cost.
        peak_power_N: Number of daily peaks for z calculation.
        p_prev: Previous power values (for MPC state).
        datetime_index_prev: Previous datetime index.
        solver: CVXPY solver.
        verbose: Print solver output.

    Returns:
        Dictionary with optimal p, q, c, d, cost, status.
    """
    T = len(load)
    q_init = q_init if q_init is not None else capacity / 2
    q_final = q_final if q_final is not None else capacity / 2

    # Variables
    p = cp.Variable(T, nonneg=True)
    c = cp.Variable(T, nonneg=True)
    d = cp.Variable(T, nonneg=True)
    q = cp.Variable(T + 1, nonneg=True)

    # Constraints
    constraints = [
        p <= MAX_GRID_POWER,
        load + c == p + d,
        q[1:] == ETA_STORE * q[:-1] + ETA_CHARGE * c - (1 / ETA_DISCHARGE) * d,
        q[0] == q_init,
        q[-1] == q_final,
        q <= capacity,
        c <= max_charge,
        d <= max_discharge,
    ]

    # Objective: energy cost
    objective = energy_cost(p, tou_prices, da_prices)

    # Add peak power cost if requested
    if include_peak_power:
        pp_cost, pp_constraints, _ = peak_power_cost(
            p, datetime_index, peak_power_N, p_prev, datetime_index_prev
        )
        objective += pp_cost
        constraints += pp_constraints

    # Solve
    problem = cp.Problem(cp.Minimize(objective), constraints)
    problem.solve(verbose=verbose, solver=solver)

    return {
        "p": p.value,
        "q": q.value,
        "c": c.value,
        "d": d.value,
        "cost": problem.value,
        "status": problem.status,
    }


def no_storage(sim: dict) -> dict:
    """Baseline: no storage, grid power equals load."""
    T = sim["T"]
    load = sim["load"]

    return {
        "p": load.copy(),
        "c": np.zeros(T),
        "d": np.zeros(T),
        "q": np.ones(T + 1) * (CAPACITY / 2),
    }


def peak_shaving(
    sim: dict,
    capacity: float = CAPACITY,
    target: float = 5.0,
) -> dict:
    """Shave peaks above target power level.

    Discharge when load exceeds target, charge when below (if battery not full).
    Target of 5 kW corresponds to tier 2 threshold.
    """
    T = sim["T"]
    load = sim["load"]
    max_charge = capacity / 2
    max_discharge = capacity / 2

    p = np.zeros(T)
    c = np.zeros(T)
    d = np.zeros(T)
    q = np.zeros(T + 1)
    q[0] = capacity / 2

    for t in range(T):
        if load[t] > target:
            needed = load[t] - target
            d[t] = min(max_discharge, needed, q[t] * ETA_DISCHARGE)
            c[t] = 0
        else:
            available = target - load[t]
            c[t] = min(max_charge, available, (capacity - q[t]) / ETA_CHARGE)
            d[t] = 0

        p[t], c[t], d[t] = _enforce_grid_limits(load[t], c[t], d[t])
        q[t + 1] = ETA_STORE * q[t] + ETA_CHARGE * c[t] - (1 / ETA_DISCHARGE) * d[t]

    return {"p": p, "c": c, "d": d, "q": q}


def energy_arbitrage(
    sim: dict,
    capacity: float = CAPACITY,
) -> dict:
    """Charge during off-peak hours, discharge during peak hours to offset load.

    Off-peak: 22:00-05:00 (charge)
    Peak: 06:00-21:00 (discharge to offset load, no export)
    """
    T = sim["T"]
    load = sim["load"]
    datetime_index = sim["datetime_index"]
    max_charge = capacity / 2
    max_discharge = capacity / 2

    p = np.zeros(T)
    c = np.zeros(T)
    d = np.zeros(T)
    q = np.zeros(T + 1)
    q[0] = capacity / 2

    for t in range(T):
        hour = datetime_index[t].hour

        if hour >= 22 or hour <= 5:
            c[t] = min(max_charge, (capacity - q[t]) / ETA_CHARGE)
            d[t] = 0
        else:
            d[t] = min(max_discharge, load[t], q[t] * ETA_DISCHARGE)
            c[t] = 0

        p[t], c[t], d[t] = _enforce_grid_limits(load[t], c[t], d[t])
        q[t + 1] = ETA_STORE * q[t] + ETA_CHARGE * c[t] - (1 / ETA_DISCHARGE) * d[t]

    return {"p": p, "c": c, "d": d, "q": q}


def mpc(
    sim: dict,
    data: dict,
    forecaster: dict,
    capacity: float = CAPACITY,
    horizon: int = HORIZON,
    include_peak_power: bool = True,
    peak_power_N: int = 3,
    solver=cp.GUROBI,
    verbose: bool = False,
    progress: bool = True,
) -> dict:
    """MPC policy: rolling horizon with forecasts.

    Args:
        sim: Simulation window dict.
        data: Full data dict from load_data().
        forecaster: Forecaster params dict.
        capacity: Storage capacity.
        horizon: Planning horizon (hours).
        include_peak_power: Whether to include peak power in objective.
        peak_power_N: Number of daily peaks (use 1 for MPC).
        solver: CVXPY solver.
        verbose: Print solver output.
        progress: Show progress bar.

    Returns:
        Dictionary with executed power flows.
    """
    T = sim["T"]
    start_idx = sim["start_idx"]
    datetime_index = sim["datetime_index"]

    load_data = data["load"]
    da_price_data = data["da_prices"]
    tou_price_data = data["tou_prices"]

    load_baseline = forecaster["load_baseline"]
    da_price_baseline = forecaster["da_price_baseline"]
    load_ar_params = forecaster["load_ar_params"]
    da_price_ar_params = forecaster["da_price_ar_params"]

    max_charge = capacity / 2
    max_discharge = capacity / 2

    # Initialize arrays
    p_mpc = np.zeros(T)
    c_mpc = np.zeros(T)
    d_mpc = np.zeros(T)
    q_mpc = np.zeros(T + 1)
    q_mpc[0] = capacity / 2

    p_prev = []
    prev_month = datetime_index[0].month

    iterator = trange(T) if progress else range(T)

    for t in iterator:
        # Reset p_prev on month boundary
        if datetime_index[t].month != prev_month:
            p_prev = []

        # Generate forecasts
        abs_t = start_idx + t
        load_forecast = make_load_forecast(
            load_data, load_baseline, load_ar_params, abs_t, horizon
        )
        da_price_forecast = make_price_forecast(
            da_price_data, da_price_baseline, da_price_ar_params, abs_t, horizon
        )
        tou_price_forecast = tou_price_data.iloc[abs_t : abs_t + horizon].values

        # Datetime indices
        horizon_index = load_baseline.index[abs_t : abs_t + horizon]
        curr_month = datetime_index[t].month
        prev_index = datetime_index[
            (datetime_index.month == curr_month) & (datetime_index < datetime_index[t])
        ]

        # Solve MPC problem
        result = optimize(
            load=load_forecast,
            tou_prices=tou_price_forecast,
            da_prices=da_price_forecast,
            datetime_index=horizon_index,
            capacity=capacity,
            max_charge=max_charge,
            max_discharge=max_discharge,
            q_init=q_mpc[t],
            q_final=capacity / 2,
            include_peak_power=include_peak_power,
            peak_power_N=peak_power_N,
            p_prev=p_prev if p_prev else None,
            datetime_index_prev=prev_index if len(prev_index) > 0 else None,
            solver=solver,
            verbose=verbose,
        )

        # Execute first action
        p_mpc[t] = result["p"][0]
        c_mpc[t] = result["c"][0]
        d_mpc[t] = result["d"][0]
        q_mpc[t + 1] = (
            ETA_STORE * q_mpc[t]
            + ETA_CHARGE * c_mpc[t]
            - (1 / ETA_DISCHARGE) * d_mpc[t]
        )

        prev_month = datetime_index[t].month
        p_prev.append(result["p"][0])

    return {
        "p": p_mpc,
        "c": c_mpc,
        "d": d_mpc,
        "q": q_mpc,
    }
