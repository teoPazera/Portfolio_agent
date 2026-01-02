# visualizations.py

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from datetime import date, datetime


def _to_1d_array(x: Sequence[float]) -> np.ndarray:
    """Safely convert jax/numpy/sequence to a 1D numpy array."""
    arr = np.asarray(x)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    return arr


def _to_x_values(x: Optional[Sequence], n: int) -> tuple[np.ndarray, bool]:
    """
    Convert an optional x-axis sequence into values usable by Matplotlib.

    Returns
    -------
    x_values : np.ndarray
        Either np.arange(n) if x is None, or converted values matching length n.
        Datetime-like inputs are converted to Matplotlib date numbers.
    is_date : bool
        True if x_values represent dates and should be formatted as such.
    """
    if x is None:
        return np.arange(n), False

    arr = np.asarray(x)
    if arr.ndim != 1:
        arr = arr.reshape(-1)

    if len(arr) != n:
        raise ValueError(f"x must have length {n}, got {len(arr)}")

    # numpy datetime64 -> python datetime -> matplotlib date numbers
    if np.issubdtype(arr.dtype, np.datetime64):
        py_dates = arr.astype("datetime64[ms]").astype("O")
        return mdates.date2num(py_dates), True

    # python datetime/date (or pandas.Timestamp, which is datetime-like)
    if arr.dtype == object and len(arr) > 0:
        first = arr[0]
        if isinstance(first, (datetime, date)):
            return mdates.date2num(arr), True

    return arr, False


def _format_date_axis(ax: plt.Axes) -> None:
    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))


def plot_loss_curve(
    train_losses: Sequence[float],
    val_losses: Optional[Sequence[float]] = None,
    show: bool = True,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot the evolution of the train (and optional validation) loss over training iterations.

    Parameters
    ----------
    train_losses : Sequence[float]
        Train loss values per iteration, e.g. from train_drl().
    val_losses : Sequence[float], optional
        Validation loss values per iteration, same length as train_losses.
        Can contain NaNs where validation was not evaluated.
    show : bool
        Whether to call plt.show() at the end.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw on. If None, a new figure+axes is created.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes with the plotted loss curves.
    """
    train_arr = _to_1d_array(train_losses)

    if ax is None:
        fig, ax = plt.subplots()

    ax.plot(np.arange(len(train_arr)), train_arr, label="Train loss")

    if val_losses is not None:
        val_arr = _to_1d_array(val_losses)
        # Mask NaNs if any (where we didn't eval val loss)
        mask = ~np.isnan(val_arr)
        ax.plot(np.arange(len(val_arr))[mask], val_arr[mask],
                label="Val loss", linestyle="--")

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title("Loss over Time")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()

    if show:
        plt.show()

    return ax


def plot_episode_performance_split(
    rewards_train: Sequence[float],
    rewards_val: Optional[Sequence[float]] = None,
    baseline_simple_train: Optional[Sequence[float]] = None,
    baseline_simple_val: Optional[Sequence[float]] = None,
    dates_train: Optional[Sequence] = None,
    dates_val: Optional[Sequence] = None,
    initial_wealth: float = 1.0,
    show: bool = True,
    label_train: str = "Train",
    label_val: str = "Val",
    label_policy: str = "RL",
) -> Tuple[plt.Figure, Optional[plt.Figure]]:
    """
    Plot TRAIN and VAL performance in separate figures.

    For each split, creates a 2x1 figure:
      - Top: per-step rewards (policy vs SP500 buy-and-hold, if provided)
      - Bottom: wealth curves (policy vs SP500 buy-and-hold)

    Assumes per-step RL rewards are:
        r_t = log(1 + portfolio_return_t) - transaction_cost_t

    For the SP500 (or SPY) buy-and-hold baseline, we assume:
        baseline_reward_t = log(1 + r_SP500_t)
        wealth_SP500_t = initial_wealth * exp(cumsum(baseline_reward_t))

    Parameters
    ----------
    rewards_train : Sequence[float]
        Per-step rewards on the training episode (policy).
    rewards_val : Sequence[float], optional
        Per-step rewards on the validation episode (policy).
    baseline_simple_train : Sequence[float], optional
        Simple returns of SP500/SPY over the training period, aligned in time
        with rewards_train.
    baseline_simple_val : Sequence[float], optional
        Simple returns of SP500/SPY over the validation period, aligned in time
        with rewards_val.
    dates_train : Sequence, optional
        Datetime-like sequence for the train x-axis, length = len(rewards_train).
        If provided, plots use dates instead of integer time steps.
    dates_val : Sequence, optional
        Datetime-like sequence for the val/test x-axis, length = len(rewards_val).
    label_train : str
        Label used for the train split in titles/legends.
    label_val : str
        Label used for the second split (val/test) in titles/legends.
    label_policy : str
        Label suffix for the policy curves (e.g. "RL", "ES").
    initial_wealth : float
        Starting wealth value for the cumulative wealth curves.
    show : bool
        Whether to call plt.show() at the end.

    Returns
    -------
    fig_train : matplotlib.figure.Figure
        Figure with train rewards + wealth.
    fig_val : matplotlib.figure.Figure or None
        Figure with val rewards + wealth, or None if rewards_val is None.
    """
    # ===== TRAIN =====
    r_train = _to_1d_array(rewards_train)
    x_train, x_train_is_date = _to_x_values(dates_train, len(r_train))
    cum_log_train = np.cumsum(r_train)
    wealth_train = initial_wealth * np.exp(cum_log_train)

    if baseline_simple_train is not None:
        b_train_simple = _to_1d_array(baseline_simple_train)
        b_train_reward = np.log1p(b_train_simple)
        if dates_train is not None and len(b_train_reward) != len(r_train):
            raise ValueError("baseline_simple_train must align with rewards_train when dates_train is provided.")
        x_b_train = x_train if len(b_train_reward) == len(r_train) else np.arange(len(b_train_reward))
        x_b_train_is_date = x_train_is_date and (len(b_train_reward) == len(r_train))
        b_train_cum_log = np.cumsum(b_train_reward)
        wealth_b_train = initial_wealth * np.exp(b_train_cum_log)
    else:
        b_train_reward = None
        x_b_train = None
        x_b_train_is_date = False
        wealth_b_train = None

    fig_train, (ax_r_tr, ax_w_tr) = plt.subplots(2, 1, figsize=(10, 6), sharex=False)

    # train rewards
    ax_r_tr.plot(x_train, r_train, label=f"{label_train} ({label_policy})")
    if b_train_reward is not None:
        ax_r_tr.plot(x_b_train, b_train_reward, alpha=0.7, label=f"{label_train} (SP500 B&H)")
    ax_r_tr.set_xlabel("Date" if x_train_is_date else "Time step")
    ax_r_tr.set_ylabel("Reward")
    ax_r_tr.set_title(f"{label_train}: Per-step Rewards")
    ax_r_tr.grid(True, linestyle="--", alpha=0.3)
    ax_r_tr.legend()

    # train wealth
    ax_w_tr.plot(x_train, wealth_train, label=f"{label_train} wealth ({label_policy})")
    if wealth_b_train is not None:
        ax_w_tr.plot(x_b_train, wealth_b_train, alpha=0.7, label=f"{label_train} wealth (SP500 B&H)")
    ax_w_tr.set_xlabel("Date" if x_train_is_date else "Time step")
    ax_w_tr.set_ylabel("Wealth")
    ax_w_tr.set_title(f"{label_train}: Wealth Curve")
    ax_w_tr.grid(True, linestyle="--", alpha=0.3)
    ax_w_tr.legend()

    if x_train_is_date or x_b_train_is_date:
        _format_date_axis(ax_r_tr)
        _format_date_axis(ax_w_tr)
        fig_train.autofmt_xdate()

    fig_train.tight_layout()

    # ===== VAL =====
    fig_val = None
    if rewards_val is not None:
        r_val = _to_1d_array(rewards_val)
        x_val, x_val_is_date = _to_x_values(dates_val, len(r_val))
        cum_log_val = np.cumsum(r_val)
        wealth_val = initial_wealth * np.exp(cum_log_val)

        if baseline_simple_val is not None:
            b_val_simple = _to_1d_array(baseline_simple_val)
            b_val_reward = np.log1p(b_val_simple)
            if dates_val is not None and len(b_val_reward) != len(r_val):
                raise ValueError("baseline_simple_val must align with rewards_val when dates_val is provided.")
            x_b_val = x_val if len(b_val_reward) == len(r_val) else np.arange(len(b_val_reward))
            x_b_val_is_date = x_val_is_date and (len(b_val_reward) == len(r_val))
            b_val_cum_log = np.cumsum(b_val_reward)
            wealth_b_val = initial_wealth * np.exp(b_val_cum_log)
        else:
            b_val_reward = None
            x_b_val = None
            x_b_val_is_date = False
            wealth_b_val = None

        fig_val, (ax_r_val, ax_w_val) = plt.subplots(2, 1, figsize=(10, 6), sharex=False)

        # val rewards
        ax_r_val.plot(x_val, r_val, label=f"{label_val} ({label_policy})")
        if b_val_reward is not None:
            ax_r_val.plot(x_b_val, b_val_reward, alpha=0.7, label=f"{label_val} (SP500 B&H)")
        ax_r_val.set_xlabel("Date" if x_val_is_date else "Time step")
        ax_r_val.set_ylabel("Reward")
        ax_r_val.set_title(f"{label_val}: Per-step Rewards")
        ax_r_val.grid(True, linestyle="--", alpha=0.3)
        ax_r_val.legend()

        # val wealth
        ax_w_val.plot(x_val, wealth_val, label=f"{label_val} wealth ({label_policy})")
        if wealth_b_val is not None:
            ax_w_val.plot(x_b_val, wealth_b_val, alpha=0.7, label=f"{label_val} wealth (SP500 B&H)")
        ax_w_val.set_xlabel("Date" if x_val_is_date else "Time step")
        ax_w_val.set_ylabel("Wealth")
        ax_w_val.set_title(f"{label_val}: Wealth Curve")
        ax_w_val.grid(True, linestyle="--", alpha=0.3)
        ax_w_val.legend()

        if x_val_is_date or x_b_val_is_date:
            _format_date_axis(ax_r_val)
            _format_date_axis(ax_w_val)
            fig_val.autofmt_xdate()

        fig_val.tight_layout()

    if show:
        plt.show()

    return fig_train, fig_val


def plot_allocation_over_time(
    weights: np.ndarray,
    asset_labels: Optional[Sequence[str]] = None,
    dates: Optional[Sequence] = None,
    title: str = "Portfolio Weights Over Time",
    show: bool = True,
) -> plt.Axes:
    """
    Plot the allocation of assets over time as a stacked area chart.

    Parameters
    ----------
    weights : np.ndarray
        Array of shape [T, N_assets] or [T, N_assets+1].
        Each row should sum (approximately) to 1.
    asset_labels : Sequence[str], optional
        Names of the assets, length N_assets. If None, generic labels are used.
    dates : Sequence, optional
        Datetime-like sequence for the x-axis, length T. If provided, the plot
        uses dates instead of integer time steps.
    title : str
        Plot title.
    show : bool
        Whether to call plt.show() at the end.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes with the plotted allocation.
    """
    w = np.asarray(weights)
    if w.ndim != 2:
        raise ValueError(f"weights must be 2D [T, N_assets], got shape {w.shape}")

    T, N_assets = w.shape
    x, x_is_date = _to_x_values(dates, T)

    # Check if rows sum to ~1
    row_sums = w.sum(axis=1)
    max_dev = np.max(np.abs(row_sums - 1.0))
    if max_dev > 1e-3:
        print(f"[plot_allocation_over_time] Warning: row sums deviate from 1 by up to {max_dev:.3e}")

    if asset_labels is None:
        asset_labels = [f"Asset {i}" for i in range(N_assets)]

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.stackplot(x, w.T, labels=asset_labels)
    ax.set_xlabel("Date" if x_is_date else "Time step")
    ax.set_ylabel("Weight")
    ax.set_title(title)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))

    fig.tight_layout()
    if x_is_date:
        _format_date_axis(ax)
        fig.autofmt_xdate()

    if show:
        plt.show()

    return ax
