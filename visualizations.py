# visualizations.py

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt


def _to_1d_array(x: Sequence[float]) -> np.ndarray:
    """Safely convert jax/numpy/sequence to a 1D numpy array."""
    arr = np.asarray(x)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    return arr


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
    initial_wealth: float = 1.0,
    show: bool = True,
) -> Tuple[plt.Figure, Optional[plt.Figure]]:
    """
    Plot TRAIN and VAL performance in separate figures.

    For each split, creates a 2x1 figure:
      - Top: per-step rewards (RL vs SP500 buy-and-hold, if provided)
      - Bottom: wealth curves (RL vs SP500 buy-and-hold)

    Assumes per-step RL rewards are:
        r_t = log(1 + portfolio_return_t) - transaction_cost_t

    For the SP500 (or SPY) buy-and-hold baseline, we assume:
        baseline_reward_t = log(1 + r_SP500_t)
        wealth_SP500_t = initial_wealth * exp(cumsum(baseline_reward_t))

    Parameters
    ----------
    rewards_train : Sequence[float]
        Per-step rewards on the training episode (RL policy).
    rewards_val : Sequence[float], optional
        Per-step rewards on the validation episode (RL policy).
    baseline_simple_train : Sequence[float], optional
        Simple returns of SP500/SPY over the training period, aligned in time
        with rewards_train.
    baseline_simple_val : Sequence[float], optional
        Simple returns of SP500/SPY over the validation period, aligned in time
        with rewards_val.
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
    t_train = np.arange(len(r_train))
    cum_log_train = np.cumsum(r_train)
    wealth_train = initial_wealth * np.exp(cum_log_train)

    if baseline_simple_train is not None:
        b_train_simple = _to_1d_array(baseline_simple_train)
        b_train_reward = np.log1p(b_train_simple)
        t_b_train = np.arange(len(b_train_reward))
        b_train_cum_log = np.cumsum(b_train_reward)
        wealth_b_train = initial_wealth * np.exp(b_train_cum_log)
    else:
        b_train_reward = None
        t_b_train = None
        wealth_b_train = None

    fig_train, (ax_r_tr, ax_w_tr) = plt.subplots(2, 1, figsize=(10, 6), sharex=False)

    # train rewards
    ax_r_tr.plot(t_train, r_train, label="Train (RL)")
    if b_train_reward is not None:
        ax_r_tr.plot(t_b_train, b_train_reward, alpha=0.7, label="Train (SP500 B&H)")
    ax_r_tr.set_xlabel("Time step")
    ax_r_tr.set_ylabel("Reward")
    ax_r_tr.set_title("Train: Per-step Rewards")
    ax_r_tr.grid(True, linestyle="--", alpha=0.3)
    ax_r_tr.legend()

    # train wealth
    ax_w_tr.plot(t_train, wealth_train, label="Train wealth (RL)")
    if wealth_b_train is not None:
        ax_w_tr.plot(t_b_train, wealth_b_train, alpha=0.7, label="Train wealth (SP500 B&H)")
    ax_w_tr.set_xlabel("Time step")
    ax_w_tr.set_ylabel("Wealth")
    ax_w_tr.set_title("Train: Wealth Curve")
    ax_w_tr.grid(True, linestyle="--", alpha=0.3)
    ax_w_tr.legend()

    fig_train.tight_layout()

    # ===== VAL =====
    fig_val = None
    if rewards_val is not None:
        r_val = _to_1d_array(rewards_val)
        t_val = np.arange(len(r_val))
        cum_log_val = np.cumsum(r_val)
        wealth_val = initial_wealth * np.exp(cum_log_val)

        if baseline_simple_val is not None:
            b_val_simple = _to_1d_array(baseline_simple_val)
            b_val_reward = np.log1p(b_val_simple)
            t_b_val = np.arange(len(b_val_reward))
            b_val_cum_log = np.cumsum(b_val_reward)
            wealth_b_val = initial_wealth * np.exp(b_val_cum_log)
        else:
            b_val_reward = None
            t_b_val = None
            wealth_b_val = None

        fig_val, (ax_r_val, ax_w_val) = plt.subplots(2, 1, figsize=(10, 6), sharex=False)

        # val rewards
        ax_r_val.plot(t_val, r_val, label="Val (RL)")
        if b_val_reward is not None:
            ax_r_val.plot(t_b_val, b_val_reward, alpha=0.7, label="Val (SP500 B&H)")
        ax_r_val.set_xlabel("Time step")
        ax_r_val.set_ylabel("Reward")
        ax_r_val.set_title("Val: Per-step Rewards")
        ax_r_val.grid(True, linestyle="--", alpha=0.3)
        ax_r_val.legend()

        # val wealth
        ax_w_val.plot(t_val, wealth_val, label="Val wealth (RL)")
        if wealth_b_val is not None:
            ax_w_val.plot(t_b_val, wealth_b_val, alpha=0.7, label="Val wealth (SP500 B&H)")
        ax_w_val.set_xlabel("Time step")
        ax_w_val.set_ylabel("Wealth")
        ax_w_val.set_title("Val: Wealth Curve")
        ax_w_val.grid(True, linestyle="--", alpha=0.3)
        ax_w_val.legend()

        fig_val.tight_layout()

    if show:
        plt.show()

    return fig_train, fig_val


def plot_allocation_over_time(
    weights: np.ndarray,
    asset_labels: Optional[Sequence[str]] = None,
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
    x = np.arange(T)

    # Check if rows sum to ~1
    row_sums = w.sum(axis=1)
    max_dev = np.max(np.abs(row_sums - 1.0))
    if max_dev > 1e-3:
        print(f"[plot_allocation_over_time] Warning: row sums deviate from 1 by up to {max_dev:.3e}")

    if asset_labels is None:
        asset_labels = [f"Asset {i}" for i in range(N_assets)]

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.stackplot(x, w.T, labels=asset_labels)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Weight")
    ax.set_title(title)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))

    fig.tight_layout()

    if show:
        plt.show()

    return ax
