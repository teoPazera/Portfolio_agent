# v1_ES.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Callable

import jax.numpy as jnp
from jax import random, vmap
from jax.flatten_util import ravel_pytree

from v1_MLP import MLP
from v1_steps import episode_loss_mixed


@dataclass
class ESConfig:
    """
    Evolution Strategies config for training the MLP policy.

    We optimize the *expected* episode loss of episode_loss_mixed using a
    simple mirrored-ES (OpenAI ES style).
    """
    num_generations: int = 200   # ES iterations
    pop_size: int = 32           # number of perturbation pairs
    sigma: float = 0.02          # noise std for parameters
    lr: float = 0.05             # learning rate on theta

    # env / objective hyperparams (match your TrainConfig)
    cost_rate: float = 1e-3
    temperature: float = 2.0
    k_rebalance: int = 15
    horizon_H: int = 100
    w_sharpe: float = 1.0
    w_return: float = 0.0
    lambda_prior: float = 0.0

    log_every: int = 10
    eval_every: int = 1
    seed: int = 0

    # elite / best-so-far tracking
    track_elite: bool = True
    elite_metric: str = "train"  # "train" or "val"


def train_es(
    mlp: MLP,
    feat_base_train: jnp.ndarray,
    asset_simple_train: jnp.ndarray,
    config: ESConfig,
    feat_base_val: Optional[jnp.ndarray] = None,
    asset_simple_val: Optional[jnp.ndarray] = None,
    prior_weights: Optional[jnp.ndarray] = None,
    loss_fn: Callable = episode_loss_mixed,
) -> tuple[MLP, jnp.ndarray, Optional[jnp.ndarray]]:
    """
    Train the MLP policy using ES on a single episode (full train period).

    Parameters
    ----------
    mlp : MLP
        Initial MLP model.
    feat_base_train : [T_train, F]
    asset_simple_train : [T_train, N_assets]
    config : ESConfig
        ES hyperparameters.
    feat_base_val, asset_simple_val : optional
        Validation set for monitoring.
    prior_weights : [N_assets+1], optional
        Baseline portfolio (e.g. 60/40 SPY/AGG) for the prior term.
    loss_fn : callable
        MLP objective with signature:
            loss_fn(params, mlp.config, feat_base, asset_simple,
                    cost_rate, temperature, k_rebalance, horizon_H,
                    w_sharpe, w_return, lambda_prior, prior_weights)
        Defaults to episode_loss_mixed.

    Returns
    -------
    mlp_trained : MLP
    train_losses : [num_generations]
    val_losses   : [num_generations] or None
    """
    theta0, unravel_fn = ravel_pytree(mlp.params)
    num_params = theta0.size

    theta = theta0
    train_losses_list = []
    val_losses_list = []

    has_val = (feat_base_val is not None) and (asset_simple_val is not None)

    best_train_loss = float("inf")
    best_train_gen = -1
    best_train_theta = theta

    best_val_loss = float("inf")
    best_val_gen = -1
    best_val_theta = None

    def loss_from_flat(theta_flat: jnp.ndarray,
                       feat_base: jnp.ndarray,
                       asset_simple: jnp.ndarray) -> jnp.ndarray:
        params = unravel_fn(theta_flat)
        return loss_fn(
            params,
            mlp.config,
            feat_base,
            asset_simple,
            config.cost_rate,
            config.temperature,
            config.k_rebalance,
            config.horizon_H,
            config.w_sharpe,
            config.w_return,
            config.lambda_prior,
            prior_weights,
        )

    key = random.PRNGKey(config.seed)

    for gen in range(config.num_generations):
        key, subkey = random.split(key)

        # eps ~ N(0, I)
        eps = random.normal(subkey, shape=(config.pop_size, num_params))

        theta_plus = theta + config.sigma * eps
        theta_minus = theta - config.sigma * eps

        def eval_candidate(theta_flat):
            return loss_from_flat(theta_flat, feat_base_train, asset_simple_train)

        losses_plus = vmap(eval_candidate)(theta_plus)   # [pop_size]
        losses_minus = vmap(eval_candidate)(theta_minus) # [pop_size]

        # gradient estimate: grad ≈ (1 / (2 N sigma)) sum_i (L+ - L-) * eps_i
        loss_diff = losses_plus - losses_minus           # [pop_size]
        grad_est = (loss_diff[:, None] * eps).sum(axis=0)
        grad_est = grad_est / (2.0 * config.pop_size * config.sigma)

        # gradient descent on theta (minimize loss)
        theta = theta - config.lr * grad_est

        center_loss = loss_from_flat(theta, feat_base_train, asset_simple_train)
        train_losses_list.append(center_loss)

        if config.track_elite:
            center_loss_f = float(center_loss)
            if center_loss_f < best_train_loss:
                best_train_loss = center_loss_f
                best_train_gen = gen
                best_train_theta = theta

        # validation monitoring
        if has_val and (config.eval_every > 0) and (gen % config.eval_every == 0):
            val_loss = loss_from_flat(theta, feat_base_val, asset_simple_val)
            val_losses_list.append(val_loss)

            if config.track_elite:
                val_loss_f = float(val_loss)
                if val_loss_f < best_val_loss:
                    best_val_loss = val_loss_f
                    best_val_gen = gen
                    best_val_theta = theta
        elif has_val:
            val_losses_list.append(jnp.nan)

        if config.log_every > 0 and (gen % config.log_every == 0):
            train_score = float(-center_loss)
            msg = f"[ES] gen {gen:4d} | train_loss={float(center_loss):.6f} | train_score={train_score:.6f}"
            if has_val:
                val_loss_disp = val_losses_list[-1]
                if not jnp.isnan(val_loss_disp):
                    msg += f" | val_loss={float(val_loss_disp):.6f} | val_score={float(-val_loss_disp):.6f}"
            print(msg)

    if config.track_elite:
        metric = config.elite_metric.strip().lower()
        use_val = has_val and metric in ("val", "eval") and (best_val_theta is not None)
        if use_val:
            theta = best_val_theta
            elite_gen = best_val_gen
            elite_loss = best_val_loss
            elite_label = "val"
        else:
            theta = best_train_theta
            elite_gen = best_train_gen
            elite_loss = best_train_loss
            elite_label = "train"

        print(
            f"[ES] elite_{elite_label} gen {elite_gen:4d} | "
            f"loss={elite_loss:.6f} | score={(-elite_loss):.6f}"
        )

    final_params = unravel_fn(theta)
    mlp_trained = MLP(config=mlp.config, params=final_params)

    train_losses = jnp.stack(train_losses_list)
    if has_val:
        val_losses = jnp.stack(val_losses_list)
    else:
        val_losses = None

    return mlp_trained, train_losses, val_losses
