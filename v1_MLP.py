# v1_MLP.py

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from jax import random, tree_util, nn


Params = dict[str, jnp.ndarray]


@dataclass
class MLPConfig:
    """Configuration of a simple one-hidden-layer MLP."""
    input_dim: int
    hidden_dim: int
    output_dim: int


@dataclass
class MLP:
    """
    One-hidden-layer MLP in JAX.

    - config: stores (input_dim, hidden_dim, output_dim)
    - params: dict with W1, b1, W2, b2
    """
    config: MLPConfig
    params: Params

    @classmethod
    def init(
        cls,
        key: random.PRNGKey,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
    ) -> "MLP":
        """
        Initialize MLP parameters with simple He-style init.
        """
        cfg = MLPConfig(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
        )

        k1, k2, k3 = random.split(key, num=3)

        # Hidden layer
        W1 = random.normal(k1, (input_dim, hidden_dim)) * jnp.sqrt(2.0 / input_dim)
        b1 = jnp.zeros((hidden_dim,))

        W2 = random.normal(k2, (hidden_dim, hidden_dim)) * jnp.sqrt(2.0 / hidden_dim)
        b2 = jnp.zeros((hidden_dim,))

        W3 = random.normal(k3, (hidden_dim, output_dim)) * jnp.sqrt(2.0 / hidden_dim)
        b3 = jnp.zeros((output_dim,))
        params: Params = {
            "W1": W1,
            "b1": b1,
            "W2": W2,
            "b2": b2,
            "W3": W3,
            "b3": b3
        }

        return cls(config=cfg, params=params)

    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass.

        Parameters
        ----------
        x : jnp.ndarray
            Input of shape [..., input_dim].

        Returns
        -------
        jnp.ndarray
            Output logits of shape [..., output_dim].
        """
        W1, b1 = self.params["W1"], self.params["b1"]
        W2, b2 = self.params["W2"], self.params["b2"]
        W3, b3 = self.params["W3"], self.params["b3"]

        # Hidden layer with tanh activation (can be changed to relu if you prefer)
        h = nn.gelu(jnp.dot(x, W1) + b1)
        h2 = nn.gelu(jnp.dot(h, W2) + b2)
        # Output layer (no activation; caller can apply softmax / temperature mapping)
        logits = jnp.dot(h2, W3) + b3
        return logits

    def apply_gradients(self, grads: Params, lr: float) -> "MLP":
        """
        Return a new MLP with parameters updated by SGD step:

            params_new = params - lr * grads

        This is handy for gradient-based RL.
        """
        new_params = tree_util.tree_map(lambda p, g: p - lr * g, self.params, grads)
        return MLP(config=self.config, params=new_params)

    def replace_params(self, new_params: Params) -> "MLP":
        """
        Return a new MLP with params replaced (useful for ES where you
        manipulate flattened vectors then unflatten back into param dicts).
        """
        return MLP(config=self.config, params=new_params)
