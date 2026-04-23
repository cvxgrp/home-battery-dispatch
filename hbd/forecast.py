"""Utility functions for building baseline-residual forecasts."""

import numpy as np
import pandas as pd
import cvxpy as cp

AR_LOOKBACK = 24  # Hours of history for AR model
AR_HORIZON = 23  # Hours to forecast with AR
N_HARMONICS = 4  # Number of Fourier harmonics per period
PERIODS = [24, 24 * 7, 24 * 365]  # Daily, weekly, annual


def featurize_baseline(t: int, n_harmonics: int = N_HARMONICS) -> list[float]:
    """Generate Fourier features for time t."""
    harmonics = []
    for P in PERIODS:
        harmonics += [P / n for n in range(1, n_harmonics + 1)]

    features = [1.0]
    for h in harmonics:
        features.append(np.sin(2 * np.pi * t / h))
        features.append(np.cos(2 * np.pi * t / h))

    return features


def train_baseline(
    train_data: pd.Series,
    eta: float = 0.5,
    lambd: float = 0.1,
) -> np.ndarray:
    """Fit seasonal baseline using quantile regression.

    Args:
        train_data: Training time series.
        eta: Quantile for pinball loss (0.5 = median).
        lambd: l2 regularization weight.

    Returns:
        Fitted coefficient vector theta.
    """
    T = len(train_data)
    X = np.array([featurize_baseline(t) for t in range(T)])
    y = train_data.values

    theta = cp.Variable(X.shape[1])
    sqrt_mu = np.array(list(range(1, N_HARMONICS + 1)) * (len(PERIODS) * 2))

    r = X @ theta - y
    pinball_loss = cp.sum(cp.maximum(eta * r, (eta - 1) * r))
    l2_reg = lambd * cp.sum_squares(cp.multiply(sqrt_mu, theta[1:]))

    problem = cp.Problem(cp.Minimize(pinball_loss + l2_reg))
    problem.solve(solver=cp.MOSEK)

    return theta.value


def featurize_residual(
    obs: np.ndarray,
    M: int,
    L: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Create feature matrix and targets for AR model.

    Args:
        obs: Residual time series.
        M: Number of past observations (lookback).
        L: Forecast horizon.

    Returns:
        X: Feature matrix of shape (n_samples, M).
        y: Target matrix of shape (n_samples, L).
    """
    n_samples = len(obs) - M - L + 1
    X = np.zeros((n_samples, M))
    y = np.zeros((n_samples, L))

    for j in range(n_samples):
        X[j, :] = obs[j : j + M]
        y[j, :] = obs[j + M : j + M + L]

    return X, y


def train_ar_model(
    residuals: np.ndarray,
    M: int = AR_LOOKBACK,
    L: int = AR_HORIZON,
    eta: float = 0.5,
    lambd: float = 0.1,
) -> np.ndarray:
    """Fit AR model for residual forecasting.

    Args:
        residuals: Residual time series (data - baseline).
        M: Number of past observations (lookback).
        L: Forecast horizon.
        eta: Quantile for pinball loss.
        lambd: l2 regularization weight.

    Returns:
        AR parameter matrix Gamma of shape (M, L).
    """
    X, y = featurize_residual(residuals, M, L)

    Gamma = cp.Variable((M, L))
    r = X @ Gamma - y
    pinball_loss = cp.sum(cp.maximum(eta * r, (eta - 1) * r))
    l2_reg = lambd * cp.sum_squares(Gamma)

    problem = cp.Problem(cp.Minimize(pinball_loss + l2_reg))
    problem.solve(solver=cp.MOSEK)

    return Gamma.value


def predict_ar(
    data: pd.Series,
    baseline: pd.Series,
    ar_params: np.ndarray,
    t: int,
    M: int,
    L: int,
    clip_min: float,
    clip_max: float,
) -> np.ndarray:
    """Generate AR forecast from time t."""
    past = data.iloc[t - M : t].values
    past_baseline = baseline.iloc[t - M : t].values
    fut_baseline = baseline.iloc[t : t + L].values

    residual_forecast = (past - past_baseline) @ ar_params
    forecast = fut_baseline + residual_forecast

    return np.clip(forecast, clip_min, clip_max)


def make_load_forecast(
    load_data: pd.Series,
    load_baseline: pd.Series,
    ar_params: np.ndarray,
    t: int,
    horizon: int,
    M: int = AR_LOOKBACK,
    L: int = AR_HORIZON,
) -> np.ndarray:
    """Generate load forecast for MPC.

    Args:
        load_data: Full load series.
        load_baseline: Full baseline series.
        ar_params: AR parameters.
        t: Current time index (absolute).
        horizon: Planning horizon.
        M: AR lookback.
        L: AR forecast horizon.

    Returns:
        Load forecast array of length horizon.
    """
    load_min, load_max = load_data.min(), load_data.max()
    curr_load = np.array([load_data.iloc[t]])

    ar_forecast = predict_ar(
        load_data, load_baseline, ar_params, t + 1, M, L, load_min, load_max
    )[: horizon - 1]

    if horizon > L:
        baseline_only = load_baseline.iloc[t + 1 + L : t + horizon].values
    else:
        baseline_only = np.array([])

    return np.concatenate([curr_load, ar_forecast, baseline_only])


def make_price_forecast(
    da_price_data: pd.Series,
    da_price_baseline: pd.Series,
    ar_params: np.ndarray,
    t: int,
    horizon: int,
    M: int = AR_LOOKBACK,
    L: int = AR_HORIZON,
) -> np.ndarray:
    """Generate day-ahead price forecast for MPC.

    Day-ahead prices are known from publication at 13:00 for next day.
    """
    price_min, price_max = da_price_data.min(), da_price_data.max()
    current_hour = da_price_data.index[t].hour

    # Hours with known prices (published at 13:00 for next day)
    if current_hour < 13:
        hours_known = 24 - current_hour
    else:
        hours_known = (24 - current_hour) + 24
    hours_known = min(hours_known, horizon)

    ar_hours = np.clip(horizon - hours_known, 0, L)
    baseline_hours = max(horizon - hours_known - ar_hours, 0)

    known_prices = da_price_data.iloc[t : t + hours_known].values

    if ar_hours > 0:
        ar_forecast = predict_ar(
            da_price_data,
            da_price_baseline,
            ar_params,
            t + hours_known,
            M,
            L,
            price_min,
            price_max,
        )[:ar_hours]
        forecast = np.concatenate([known_prices, ar_forecast])
    else:
        forecast = known_prices

    if baseline_hours > 0:
        baseline_only = da_price_baseline.iloc[
            t + hours_known + ar_hours : t + horizon
        ].values
        forecast = np.concatenate([forecast, baseline_only])

    return forecast
