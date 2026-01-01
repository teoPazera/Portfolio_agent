# Final Project: Portfolio Allocation (DRL vs ES)

This project builds a small, self-contained portfolio-allocation “environment” and trains the same JAX MLP policy with two optimizers:
- **DRL**: gradient-based optimization of a differentiable episode objective.
- **ES**: mirrored Evolution Strategies over the same objective.

The main experiment is a **sweep over `(w_sharpe, w_return)`** to see how the objective’s Sharpe-vs-return weighting changes the learned policy.

## Repository Layout
- `data/`: input CSVs (prices + returns) produced by the download notebook.
- `runs_data/`: saved experiment artifacts (gitignored; created by the sweep notebook).
- `data_download.ipynb`: downloads ETF prices via `yfinance` and writes `data/*.csv`.
- `v1_attempt.ipynb`: the “single run” notebook used for development/debugging.
- `sharpe_return_sweep.ipynb`: runs the weight sweep, makes plots, and saves outputs to `runs_data/`.
- `v1_MLP.py`, `v1_steps.py`, `v1_DRL.py`, `v1_evolutionary_strategy.py`: core implementation.
- `visualizations.py`: plotting helpers used by notebooks.

## Design Choices (Why It’s Built This Way)
- **Weights as logits + softmax (+ temperature)**: the policy outputs unconstrained logits and `softmax` maps them to valid portfolio weights that sum to 1; temperature controls how concentrated allocations can become.
- **Explicit cash asset**: the last weight is “cash” with zero return; helps the policy express “risk-off” allocations without shorting.
- **Log-return reward + transaction costs**: reward uses `log(1 + r_portfolio)` and subtracts a turnover-based cost; log returns make wealth curves additive over time and stabilize optimization.
- **Rebalancing cadence (`k_rebalance`)**: the policy only pays turnover costs when it actually changes weights, reducing unrealistic day-to-day churn.
- **Horizon objective (`horizon_H`)**: an optional “hold for H days” reward reduces noise vs single-day rewards and better matches rebalancing decisions.
- **Mixed objective**: training minimizes `loss = -(w_sharpe * Sharpe_ann + w_return * mean_ann - lambda_prior * prior_penalty)` so you can smoothly trade off “risk-adjusted” vs “raw return”.
- **Optional prior regularization**: `prior_penalty` nudges allocations toward a baseline portfolio (e.g., 60/40) to reduce extreme weights when desired.
- **JAX everywhere**: rollouts and objectives are written in JAX (`lax.scan`, `vmap`) so DRL can differentiate through the episode, and ES can batch-evaluate candidates efficiently.

## Module / Function Reference

### `v1_MLP.py`
- `MLPConfig`: network shape (`input_dim`, `hidden_dim`, `output_dim`).
- `MLP.init(key, input_dim, hidden_dim, output_dim)`: He-style init for a 2-hidden-layer MLP policy.
- `MLP.forward(x)`: returns logits for `N_assets + 1` weights given `x = [features, prev_weights]`.
- `MLP.apply_gradients(grads, lr)`: SGD update (`params - lr * grads`) used by DRL.
- `MLP.replace_params(new_params)`: convenience for swapping params (used after ES unflattens).

### `v1_steps.py` (Environment + Objectives)
- `logits_to_weights_temperature(logits, temperature)`: stable softmax mapping to valid weights.
- `compute_step_reward(w_prev, logits_t, asset_simple_returns_t, cost_rate, temperature)`: 1-day reward (log portfolio return minus turnover cost) + updated weights.
- `compute_step_reward_horizon(w_prev, logits_t, asset_simple_H_t, cost_rate, temperature)`: decision reward for “pick weights, hold for H days”.
- `compute_horizon_asset_returns(asset_simple, starts, H)`: vectorized H-day simple returns per asset for each decision start.
- `rollout_episode(params, config, feat_base, asset_simple, cost_rate, temperature, k_rebalance, horizon_H)`: runs an episode and returns per-step rewards (daily or per-decision) plus total return.
- `episode_loss(params, config, ...)`: negative Sharpe-like objective on rollout rewards (kept as a simpler baseline).
- `episode_loss_mixed(params, config, ..., w_sharpe, w_return, lambda_prior, prior_weights)`: main objective used by both DRL and ES (annualized Sharpe + annualized mean return, optional prior term).
- `rollout_episode_with_weights(params, config, ...)`: like `rollout_episode` but also returns the weight path (needed for the prior term).
- `rollout_episode_daily_eval_with_weights(params, config, ...)`: daily rollout used for plotting/benchmarking even if training used a horizon objective.

### `v1_DRL.py` (Gradient-Based Training)
- `TrainConfig`: hyperparameters for DRL (SGD steps, env knobs, mixed-objective weights, elite tracking).
- `train_drl(mlp, feat_base_train, asset_simple_train, config, feat_base_val, asset_simple_val, prior_weights)`: optimizes `episode_loss_mixed` with `value_and_grad`, optionally tracks an “elite” parameter set by train/val loss.
- `evaluate_policy(mlp, feat_base, asset_simple, ...)`: returns rewards/total return via `rollout_episode` (supports horizon mode).
- `evaluate_policy_with_weights(mlp, feat_base, asset_simple, ...)`: daily evaluation rollout that returns weights (used for plots and sweep summaries).

### `v1_evolutionary_strategy.py` (Mirrored ES)
- `ESConfig`: ES hyperparameters (population, sigma, lr) + env/objective knobs + elite tracking.
- `train_es(mlp, feat_base_train, asset_simple_train, config, feat_base_val, asset_simple_val, prior_weights, loss_fn)`: mirrored-ES update on flattened parameters using `(L+ - L-) * eps` gradient estimates.

### `visualizations.py`
- `_to_1d_array(x)`: normalizes arrays/sequences for plotting.
- `_to_x_values(x, n)`: builds x-axis values (integers or date axis via `matplotlib.dates`).
- `_format_date_axis(ax)`: applies a concise date formatter for time-series plots.
- `plot_loss_curve(train_losses, val_losses, ...)`: train/val loss history.
- `plot_episode_performance_split(rewards_train, rewards_val, baseline_simple_train, baseline_simple_val, dates_train, dates_val, ...)`: reward series + wealth curves for train/test splits (optionally vs SPY baseline).
- `plot_allocation_over_time(weights, asset_labels, dates, ...)`: stacked allocation chart over time.

## Sweep Artifacts (What Gets Saved)
`sharpe_return_sweep.ipynb` writes a timestamped folder under `runs_data/` containing:
- `summary.csv` / `summary.json`: one row per (algo, weight-pair) with key metrics.
- `avg_alloc.csv`: average test allocation per run (for quick inspection).
- `*/arrays.npz`, `*/meta.json`: per-run arrays (losses/rewards/weights) + metadata, plus `mlp_params.npz` for the learned policy.

