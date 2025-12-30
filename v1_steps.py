# v1_steps.py

from __future__ import annotations

import jax.numpy as jnp
from jax import lax
from jax import vmap

from v1_MLP import MLP, Params, MLPConfig 


def logits_to_weights_temperature(logits: jnp.ndarray, temperature: float = 2.0) -> jnp.ndarray:
    """Map raw logits to portfolio weights using temperature-scaled softmax."""
    z = logits / temperature
    z = z - jnp.max(z)          # numerical stability
    w = jnp.exp(z)
    w = w / jnp.sum(w)
    return w


def compute_step_reward(
    w_prev: jnp.ndarray,
    logits_t: jnp.ndarray,
    asset_simple_returns_t: jnp.ndarray,
    cost_rate: float = 1e-3,
    temperature: float = 2.0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute reward and new weights for a single *daily* time step.

    Parameters
    ----------
    w_prev : jnp.ndarray
        Portfolio weights at t-1, shape [N_assets + 1] (last = cash).
    logits_t : jnp.ndarray
        Raw network outputs at time t, shape [N_assets + 1].
    asset_simple_returns_t : jnp.ndarray
        Simple returns of the N_assets at day t, shape [N_assets].
    cost_rate : float
        Transaction cost rate per unit turnover.
    temperature : float
        Temperature for softmax mapping.

    Returns
    -------
    reward_t : jnp.ndarray
        Scalar reward at time t.
    w_t : jnp.ndarray
        New portfolio weights at time t, shape [N_assets + 1].
    """
    # new weights from logits
    w_t = logits_to_weights_temperature(logits_t, temperature=temperature)

    # portfolio simple return using previous weights (cash return = 0)
    w_assets_prev = w_prev[:-1]  # ignore cash for return calc
    r_p = jnp.dot(w_assets_prev, asset_simple_returns_t)  # scalar

    # log portfolio return
    log_ret_p = jnp.log1p(r_p)  # log(1 + r_p)

    # transaction cost based on turnover (L1 change)
    turnover = 0.5 * jnp.sum(jnp.abs(w_t - w_prev))
    cost = cost_rate * turnover

    reward_t = log_ret_p - cost
    return reward_t, w_t


def compute_step_reward_horizon(
    w_prev: jnp.ndarray,
    logits_t: jnp.ndarray,
    asset_simple_H_t: jnp.ndarray,  # H-day simple returns per asset, shape [N_assets]
    cost_rate: float = 1e-3,
    temperature: float = 2.0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Reward for one *decision*: pick w_t, hold for H days.

    asset_simple_H_t is the H-day simple return for each asset:
        asset_simple_H_t[i] = prod_{h=0..H-1} (1 + r_{t+h, i}) - 1
    """
    # new weights from logits
    w_t = logits_to_weights_temperature(logits_t, temperature=temperature)

    # portfolio H-day simple return if we hold w_t
    # (cash has 0 return, so ignore last weight)
    r_p_H = jnp.dot(w_t[:-1], asset_simple_H_t)
    log_ret_p_H = jnp.log1p(r_p_H)

    # transaction cost based on turnover
    turnover = 0.5 * jnp.sum(jnp.abs(w_t - w_prev))
    cost = cost_rate * turnover

    reward_t = log_ret_p_H - cost
    return reward_t, w_t


def compute_horizon_asset_returns(
    asset_simple: jnp.ndarray,   # [T, N_assets], daily simple returns
    starts: jnp.ndarray,         # [D] decision start indices (ints)
    H: int,
) -> jnp.ndarray:
    """For each start index s in `starts`, compute H-day simple returns
    for each asset: prod_{h=0..H-1} (1 + r_{s+h}) - 1.

    Returns
    -------
    jnp.ndarray
        Array of shape [D, N_assets] with H-day simple returns.
    """

    def horizon_return_for_start(s) -> jnp.ndarray:
        # slice [H, N_assets] starting at time s
        window = lax.dynamic_slice_in_dim(asset_simple, s, H, axis=0)
        # use logs for numerical stability: sum log(1 + r) then expm1
        log_r = jnp.log1p(window)                  # [H, N_assets]
        sum_log = jnp.sum(log_r, axis=0)           # [N_assets]
        return jnp.expm1(sum_log)                  # simple H-day returns

    return vmap(horizon_return_for_start)(starts)  # [D, N_assets]


def rollout_episode(
    params: Params,
    config: MLPConfig,
    feat_base: jnp.ndarray,
    asset_simple: jnp.ndarray,
    cost_rate: float = 1e-3,
    temperature: float = 2.0,
    k_rebalance: int = 15,
    horizon_H: int | None = 100,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Roll out one episode and return rewards and total return.

    Two modes:
    - If `horizon_H is None`: daily objective with rebalancing every
      `k_rebalance` steps (same behaviour as your previous code).
    - If `horizon_H` is an integer >= 1: decisions are spaced by
      `k_rebalance` days and each decision's reward is based on
      `horizon_H`-day performance of the chosen weights.

    Returns
    -------
    rewards : jnp.ndarray
        If horizon_H is None: shape [T] (per day).
        If horizon_H is not None: shape [D] (per decision).
    total_return : jnp.ndarray
        Sum of rewards over the episode (scalar).
    """
    T, F = feat_base.shape
    N_assets = asset_simple.shape[1]
    num_weights = N_assets + 1  # +1 for cash

    # -------- Horizon-based objective --------
    if horizon_H is not None:
        # We use an end-of-day decision convention: at decision time t we observe `feat_base[t]`,
        # choose new weights, and evaluate their performance over the *next* H days, i.e.
        # returns from (t+1) .. (t+H).
        if horizon_H >= T:
            raise ValueError("horizon_H must be smaller than the episode length T.")

        # valid decision time t: we need (t + H) < T  =>  t <= T - H - 1
        max_start = T - horizon_H - 1
        # number of decision steps
        num_decisions = (max_start // k_rebalance) + 1
        # decision times: 0, K, 2K, ..., <= max_start
        starts = jnp.arange(num_decisions) * k_rebalance

        # features at decision times
        feat_dec = feat_base[starts]  # [D, F]

        # H-day asset returns starting at t+1 for each decision time t
        asset_H = compute_horizon_asset_returns(asset_simple, starts + 1, horizon_H)  # [D, N_assets]

        # initial portfolio: all cash
        w0 = jnp.zeros((num_weights,))
        w0 = w0.at[-1].set(1.0)

        # rebuild MLP
        mlp = MLP(config=config, params=params)

        def scan_step_horizon(
            w_prev: jnp.ndarray,
            inputs: tuple[jnp.ndarray, jnp.ndarray],
        ) -> tuple[jnp.ndarray, jnp.ndarray]:
            feat_t, asset_H_t = inputs

            # state = [features at start, previous weights]
            state_t = jnp.concatenate([feat_t, w_prev], axis=0)
            logits_t = mlp.forward(state_t)  # [num_weights]

            reward_t, w_t = compute_step_reward_horizon(
                w_prev,
                logits_t,
                asset_H_t,
                cost_rate=cost_rate,
                temperature=temperature,
            )
            return w_t, reward_t

        _, rewards = lax.scan(
            scan_step_horizon,
            w0,
            (feat_dec, asset_H),
        )

        total_return = jnp.sum(rewards)
        return rewards, total_return

    # -------- Daily objective with k-step rebalancing --------

    # initial portfolio: all cash
    w0 = jnp.zeros((num_weights,))
    w0 = w0.at[-1].set(1.0)

    # rebuild MLP from params + config
    mlp = MLP(config=config, params=params)

    def scan_step(
        carry: tuple[jnp.ndarray],
        inputs: tuple[jnp.ndarray, jnp.ndarray],
    ) -> tuple[tuple[jnp.ndarray], jnp.ndarray]:
        w_prev, t = carry
        feat_t, r_simple_t = inputs

        state_t = jnp.concatenate([feat_t, w_prev], axis=0)
        logits_t = mlp.forward(state_t)

        def do_rebalance_fn(_):
            reward_rebal, w_rebal = compute_step_reward(
                w_prev,
                logits_t,
                r_simple_t,
                cost_rate=cost_rate,
                temperature=temperature,
            )
            return reward_rebal, w_rebal

        def hold_fn(_):
            # hold previous weights: no transaction cost
            r_p_hold = jnp.dot(w_prev[:-1], r_simple_t)
            reward_hold = jnp.log1p(r_p_hold)
            return reward_hold, w_prev

        do_rebal = (t % k_rebalance) == 0

        reward_t, w_t = lax.cond(
            do_rebal,
            do_rebalance_fn,
            hold_fn,
            operand=None,
        )

        # next carry: updated weights and incremented time
        next_carry = (w_t, t + 1)
        return next_carry, reward_t

    # initial time index = 0
    init_carry = (w0, jnp.int32(0))

    (_, _), rewards = lax.scan(
        scan_step,
        init_carry,
        (feat_base, asset_simple),
    )

    total_return = jnp.sum(rewards)
    return rewards, total_return


def episode_loss(
    params: Params,
    config: MLPConfig,
    feat_base: jnp.ndarray,
    asset_simple: jnp.ndarray,
    cost_rate: float = 1e-3,
    temperature: float = 2.0,
    k_rebalance: int = 15,
    horizon_H: int | None = 100,
) -> jnp.ndarray:
    """
    Loss for one episode: negative Sharpe-like objective.

    We first roll out the episode to get the per-step rewards, which are
    log portfolio returns net of transaction costs (daily or horizon-
    based, depending on `horizon_H`). Then we compute:

        mean_r = mean(rewards)
        std_r  = std(rewards)
        sharpe_like = mean_r / (std_r + eps)

    and return -sharpe_like as the loss.

    Notes
    -----
    - Because transaction costs are already subtracted inside
      compute_step_reward / compute_step_reward_horizon, the Sharpe-like
      objective automatically includes a penalty for over-trading.
    - This works both for:
        * daily objective (horizon_H is None → rewards per day)
        * horizon-based objective (horizon_H is int → rewards per decision)
    """
    rewards, _ = rollout_episode(
        params,
        config,
        feat_base,
        asset_simple,
        cost_rate=cost_rate,
        temperature=temperature,
        k_rebalance=k_rebalance,
        horizon_H=horizon_H,
    )

    mean_r = jnp.mean(rewards)
    std_r = jnp.std(rewards) + 1e-8  # small epsilon for numerical stability

    sharpe_like = mean_r / std_r
    return -sharpe_like

def episode_loss_mixed(
    params: Params,
    config: MLPConfig,
    feat_base: jnp.ndarray,
    asset_simple: jnp.ndarray,
    cost_rate: float = 1e-3,
    temperature: float = 2.0,
    k_rebalance: int = 15,
    horizon_H: int | None = 100,
    w_sharpe: float = 1.0,
    w_return: float = 0.0,
    lambda_prior: float = 0.0,
    prior_weights: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """
    Mixed objective for one episode:
        score = w_sharpe * Sharpe_ann + w_return * mean_ann
                - lambda_prior * E[||w_t - prior_weights||^2]

    Loss = -score (because we minimize).

    - rewards are per-step log(1 + r_p) - cost, from rollout_episode_with_weights.
    - We convert them back to approximate simple net returns via expm1
      to compute Sharpe and mean.

    Parameters
    ----------
    params, config : MLP params/config
    feat_base : [T, F]
    asset_simple : [T, N_assets]
    cost_rate, temperature, k_rebalance, horizon_H : as before
    w_sharpe : weight on annualized Sharpe term
    w_return : weight on annualized mean return
    lambda_prior : strength of prior regularization
    prior_weights : [N_assets + 1] baseline portfolio (e.g. 60/40 SPY/AGG)

    Returns
    -------
    loss : scalar (negative objective)
    """
    # rollout to get rewards and weights
    rewards, total_return, weights = rollout_episode_with_weights(
        params,
        config,
        feat_base,
        asset_simple,
        cost_rate=cost_rate,
        temperature=temperature,
        k_rebalance=k_rebalance,
        horizon_H=horizon_H,
    )

    # rewards are log(1 + r_p) - cost ≈ log(1 + r_net)
    # convert to approximate simple net returns
    r_net = jnp.expm1(rewards)  # shape [T] or [D], depending on horizon

    mean_r = jnp.mean(r_net)
    std_r = jnp.std(r_net) + 1e-8

    # annualize (assuming 252 trading days)
    # - daily objective (horizon_H is None): r_net is ~daily net simple return
    # - horizon objective: r_net is an H-day net simple return, so scale by 252 / H
    ann_factor = 252.0 if horizon_H is None else (252.0 / float(horizon_H))
    mean_ann = mean_r * ann_factor
    std_ann = std_r * jnp.sqrt(ann_factor)
    sharpe_ann = mean_ann / (std_ann + 1e-8)

    # main mixed objective
    score = w_sharpe * sharpe_ann + w_return * mean_ann

    # add prior regularization if requested
    if (lambda_prior > 0.0) and (prior_weights is not None):
        # weights: [T_or_D, N_assets+1]
        # prior_weights: [N_assets+1]
        diff = weights - prior_weights
        # mean squared distance over time
        prior_penalty = jnp.mean(jnp.sum(diff * diff, axis=1))
        score = score - lambda_prior * prior_penalty

    # we minimize loss = -score
    return -score


def rollout_episode_with_weights(
    params: Params,
    config: MLPConfig,
    feat_base: jnp.ndarray,
    asset_simple: jnp.ndarray,
    cost_rate: float = 1e-3,
    temperature: float = 2.0,
    k_rebalance: int = 15,
    horizon_H: int | None = 100,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Roll out one full episode and return rewards, total return and weights.

    If horizon_H is None:
        - rewards has shape [T] (per day),
        - weights has shape [T, N_assets + 1] (per day).

    If horizon_H is not None:
        - rewards has shape [D] (per decision),
        - weights has shape [D, N_assets + 1] (weights at decision times).
    """
    T, F = feat_base.shape
    N_assets = asset_simple.shape[1]
    num_weights = N_assets + 1  # +1 for cash

    # -------- Horizon-based objective with decision-level weights --------
    if horizon_H is not None:
        # Same convention as rollout_episode(): a decision at time t uses `feat_base[t]` and is
        # evaluated on returns from (t+1) .. (t+H).
        if horizon_H >= T:
            raise ValueError("horizon_H must be smaller than the episode length T.")

        max_start = T - horizon_H - 1
        num_decisions = (max_start // k_rebalance) + 1
        starts = jnp.arange(num_decisions) * k_rebalance

        feat_dec = feat_base[starts]  # [D, F]
        asset_H = compute_horizon_asset_returns(asset_simple, starts + 1, horizon_H)  # [D, N_assets]

        # initial portfolio: all cash
        w0 = jnp.zeros((num_weights,))
        w0 = w0.at[-1].set(1.0)

        mlp = MLP(config=config, params=params)

        def scan_step_horizon(
            w_prev: jnp.ndarray,
            inputs: tuple[jnp.ndarray, jnp.ndarray],
        ) -> tuple[jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray]]:
            feat_t, asset_H_t = inputs

            state_t = jnp.concatenate([feat_t, w_prev], axis=0)
            logits_t = mlp.forward(state_t)

            reward_t, w_t = compute_step_reward_horizon(
                w_prev,
                logits_t,
                asset_H_t,
                cost_rate=cost_rate,
                temperature=temperature,
            )
            return w_t, (reward_t, w_t)

        (w_T), (rewards, weights) = lax.scan(
            scan_step_horizon,
            w0,
            (feat_dec, asset_H),
        )

        total_return = jnp.sum(rewards)
        return rewards, total_return, weights

    # -------- Daily objective with per-day weights --------

    # initial portfolio: all cash
    w0 = jnp.zeros((num_weights,))
    w0 = w0.at[-1].set(1.0)
    mlp = MLP(config=config, params=params)

    def scan_step(
        carry: tuple[jnp.ndarray],
        inputs: tuple[jnp.ndarray, jnp.ndarray],
    ) -> tuple[tuple[jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]]:
        w_prev, t = carry
        feat_t, r_simple_t = inputs

        state_t = jnp.concatenate([feat_t, w_prev], axis=0)
        logits_t = mlp.forward(state_t)

        def do_rebalance_fn(_):
            reward_rebal, w_rebal = compute_step_reward(
                w_prev,
                logits_t,
                r_simple_t,
                cost_rate=cost_rate,
                temperature=temperature,
            )
            return reward_rebal, w_rebal

        def hold_fn(_):
            r_p_hold = jnp.dot(w_prev[:-1], r_simple_t)
            reward_hold = jnp.log1p(r_p_hold)
            return reward_hold, w_prev

        do_rebal = (t % k_rebalance) == 0
        reward_t, w_t = lax.cond(
            do_rebal,
            do_rebalance_fn,
            hold_fn,
            operand=None,
        )

        return (w_t, t + 1), (reward_t, w_t)

    (w_T, _), (rewards, weights) = lax.scan(
        scan_step,
        (w0, jnp.int32(0)),
        (feat_base, asset_simple),
    )
    total_return = jnp.sum(rewards)
    return rewards, total_return, weights



def rollout_episode_daily_eval_with_weights(
    params: Params,
    config: MLPConfig,
    feat_base: jnp.ndarray,
    asset_simple: jnp.ndarray,
    cost_rate: float = 1e-3,
    temperature: float = 2.0,
    k_rebalance: int = 15,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Daily evaluation rollout (no horizon aggregation), used only for
    plotting / benchmarking.

    - Rebalances at most every `k_rebalance` days.
    - Computes *daily* log returns (minus transaction cost on rebalance days).
    - Returns per-day rewards and weights so you can build a wealth curve.

    Returns
    -------
    rewards : [T] daily log returns (net of costs)
    total_return : scalar (sum of rewards)
    weights : [T, N_assets + 1] daily portfolio weights
    """
    T, F = feat_base.shape
    N_assets = asset_simple.shape[1]
    num_weights = N_assets + 1  # +1 for cash

    # start fully in cash
    w0 = jnp.zeros((num_weights,))
    w0 = w0.at[-1].set(1.0)

    mlp = MLP(config=config, params=params)

    def scan_step(carry, inputs):
        w_prev, t = carry
        feat_t, r_simple_t = inputs

        state_t = jnp.concatenate([feat_t, w_prev], axis=0)
        logits_t = mlp.forward(state_t)

        def do_rebalance_fn(_):
            # same as your daily objective
            reward_rebal, w_rebal = compute_step_reward(
                w_prev,
                logits_t,
                r_simple_t,
                cost_rate=cost_rate,
                temperature=temperature,
            )
            return reward_rebal, w_rebal

        def hold_fn(_):
            # hold weights, no transaction cost
            r_p_hold = jnp.dot(w_prev[:-1], r_simple_t)
            reward_hold = jnp.log1p(r_p_hold)
            return reward_hold, w_prev

        do_rebal = (t % k_rebalance) == 0

        reward_t, w_t = lax.cond(
            do_rebal,
            do_rebalance_fn,
            hold_fn,
            operand=None,
        )

        return (w_t, t + 1), (reward_t, w_t)

    (w_T, _), (rewards, weights) = lax.scan(
        scan_step,
        (w0, jnp.int32(0)),
        (feat_base, asset_simple),
    )

    total_return = jnp.sum(rewards)
    return rewards, total_return, weights
