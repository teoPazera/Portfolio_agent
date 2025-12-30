# v1_DRL.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import jax.numpy as jnp
from jax import value_and_grad

from v1_MLP import MLP
from v1_steps import episode_loss, rollout_episode, rollout_episode_with_weights, rollout_episode_daily_eval_with_weights, episode_loss_mixed

@dataclass
class TrainConfig:
    """
    Configuration for DRL policy training with simple SGD.
    """
    num_iters: int = 500
    lr: float = 1e-3
    cost_rate: float = 1e-3
    temperature: float = 2.0
    log_every: int = 50
    eval_every: int = 1
    k_rebalance: int = 15
    horizon_H: int = 100

    w_sharpe: float = 1.0
    w_return: float = 0.0
    lambda_prior: float = 0.0
    # prior_weights stays outside TrainConfig in the example below

    # elite / best-so-far tracking
    track_elite: bool = True
    elite_metric: str = "train"  # "train" or "val"


def train_drl(
    mlp: MLP,
    feat_base_train: jnp.ndarray,
    asset_simple_train: jnp.ndarray,
    config: TrainConfig,
    feat_base_val: Optional[jnp.ndarray] = None,
    asset_simple_val: Optional[jnp.ndarray] = None,
    prior_weights: Optional[jnp.ndarray] = None,
) -> tuple[MLP, jnp.ndarray, Optional[jnp.ndarray]]:


    """
    Train the MLP policy using gradient-based RL on a single episode
    (full train period), and optionally track validation loss.

    Parameters
    ----------
    mlp : MLP
        Initial MLP model (with config + params).
    feat_base_train : jnp.ndarray
        Training features, shape [T_train, F].
    asset_simple_train : jnp.ndarray
        Training simple returns, shape [T_train, N_assets].
    config : TrainConfig
        Hyperparameter configuration.
    feat_base_val : jnp.ndarray, optional
        Validation features, shape [T_val, F].
    asset_simple_val : jnp.ndarray, optional
        Validation simple returns, shape [T_val, N_assets].

    Returns
    -------
    mlp : MLP
        Trained MLP model.
    train_losses : jnp.ndarray
        Train loss history over iterations, shape [num_iters].
    val_losses : jnp.ndarray or None
        Validation loss history over iterations (NaN where not evaluated),
        or None if no validation data was provided.
    """
    train_losses_list = []
    val_losses_list = []

    has_val = (feat_base_val is not None) and (asset_simple_val is not None)
    prior_w = prior_weights

    best_train_loss = float("inf")
    best_train_iter = -1
    best_train_params = mlp.params

    best_val_loss = float("inf")
    best_val_iter = -1
    best_val_params = None

    for it in range(config.num_iters):
        # ---- train loss + grads ----
        train_loss, grads = value_and_grad(episode_loss_mixed)(
            mlp.params,
            mlp.config,
            feat_base_train,
            asset_simple_train,
            config.cost_rate,
            config.temperature,
            config.k_rebalance,
            config.horizon_H,
            config.w_sharpe,
            config.w_return,
            config.lambda_prior,
            prior_w,
        )

        if config.track_elite:
            train_loss_f = float(train_loss)
            if train_loss_f < best_train_loss:
                best_train_loss = train_loss_f
                best_train_iter = it
                best_train_params = mlp.params

        mlp = mlp.apply_gradients(grads, config.lr)
        train_losses_list.append(train_loss)

        # ---- optional validation loss ----
        if has_val and (config.eval_every > 0) and (it % config.eval_every == 0):
            val_loss = episode_loss_mixed(
                mlp.params,
                mlp.config,
                feat_base_val,
                asset_simple_val,
                config.cost_rate,
                config.temperature,
                config.k_rebalance,
                config.horizon_H,
                config.w_sharpe,
                config.w_return,
                config.lambda_prior,
                prior_w,
            )
            val_losses_list.append(val_loss)

            if config.track_elite:
                val_loss_f = float(val_loss)
                if val_loss_f < best_val_loss:
                    best_val_loss = val_loss_f
                    best_val_iter = it
                    best_val_params = mlp.params
        elif has_val:
            # keep alignment in length by appending NaN when not evaluated
            val_losses_list.append(jnp.nan)

        # ---- logging ----
        if config.log_every > 0 and (it % config.log_every == 0):
            train_score = float(-train_loss)
            log_msg = f"[DRL] iter {it:4d} | train_loss={float(train_loss):.6f} | train_score={train_score:.6f}"
            if has_val:
                val_loss_display = val_losses_list[-1]
                if not jnp.isnan(val_loss_display):
                    val_score = float(-val_loss_display)
                    log_msg += f" | val_loss={float(val_loss_display):.6f} | val_score={val_score:.6f}"
            print(log_msg)



    train_losses = jnp.stack(train_losses_list)
    if has_val:
        val_losses = jnp.stack(val_losses_list)
    else:
        val_losses = None

    if config.track_elite:
        metric = config.elite_metric.strip().lower()
        use_val = has_val and metric in ("val", "eval") and (best_val_params is not None)
        if use_val:
            elite_params = best_val_params
            elite_iter = best_val_iter
            elite_loss = best_val_loss
            elite_label = "val"
        else:
            elite_params = best_train_params
            elite_iter = best_train_iter
            elite_loss = best_train_loss
            elite_label = "train"

        print(
            f"[DRL] elite_{elite_label} iter {elite_iter:4d} | "
            f"loss={elite_loss:.6f} | score={(-elite_loss):.6f}"
        )
        mlp = MLP(config=mlp.config, params=elite_params)

    return mlp, train_losses, val_losses


def evaluate_policy(
    mlp: MLP,
    feat_base: jnp.ndarray,
    asset_simple: jnp.ndarray,
    cost_rate: float = 1e-3,
    temperature: float = 2.0,
    k_rebalance: int = 15,
    horizon_H: int = 100
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Evaluate a trained policy on a given dataset (e.g. train/val/test).

    Parameters
    ----------
    mlp : MLP
        Trained MLP model.
    feat_base : jnp.ndarray
        Features over the evaluation period, shape [T, F].
    asset_simple : jnp.ndarray
        Simple returns over the evaluation period, shape [T, N_assets].
    cost_rate : float
        Transaction cost rate.
    temperature : float
        Softmax temperature.

    Returns
    -------
    rewards : jnp.ndarray
        Per-step rewards, shape [T].
    total_return : jnp.ndarray
        Sum of rewards (scalar).
    """
    rewards, total_return = rollout_episode(
        mlp.params,
        mlp.config,
        feat_base,
        asset_simple,
        cost_rate=cost_rate,
        temperature=temperature,
        k_rebalance=k_rebalance,
        horizon_H=horizon_H
    )
    return rewards, total_return






def evaluate_policy_with_weights(mlp: MLP, feat_base: jnp.ndarray,
    asset_simple: jnp.ndarray,  cost_rate: float = 1e-3,
    temperature: float = 2.0,  k_rebalance: int = 15):
    """
    Evaluate a trained policy and also return weights over time.

    Returns
    -------
    rewards : [T]
    total_return : scalar
    weights : [T, N_assets+1]
    """
    rewards, total_return, weights = rollout_episode_daily_eval_with_weights(
        mlp.params,
        mlp.config,
        feat_base,
        asset_simple,
        cost_rate=cost_rate,
        temperature=temperature,
        k_rebalance=k_rebalance,
        # horizon_H=horizon_H
        )
 
    return rewards, total_return, weights
