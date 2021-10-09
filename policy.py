import functools
from typing import Optional, Sequence, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions
tfb = tfp.bijectors

from common import MLP, Params, PRNGKey, default_init

LOG_STD_MIN = -10.0
LOG_STD_MAX = 2.0


class NormalTanhPolicy(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    state_dependent_std: bool = True
    dropout_rate: Optional[float] = None
    log_std_scale: float = 1.0
    log_std_min: Optional[float] = None
    log_std_max: Optional[float] = None
    tanh_squash_distribution: bool = True

    @nn.compact
    def __call__(self,
                 observations: jnp.ndarray,
                 temperature: float = 1.0,
                 training: bool = False) -> tfd.Distribution:
        outputs = MLP(self.hidden_dims,
                      activate_final=True,
                      dropout_rate=self.dropout_rate)(observations,
                                                      training=training)

        means = nn.Dense(self.action_dim, kernel_init=default_init())(outputs)

        if self.state_dependent_std:
            log_stds = nn.Dense(self.action_dim,
                                kernel_init=default_init(
                                    self.log_std_scale))(outputs)
        else:
            log_stds = self.param('log_stds', nn.initializers.zeros,
                                  (self.action_dim, ))

        log_std_min = self.log_std_min or LOG_STD_MIN
        log_std_max = self.log_std_max or LOG_STD_MAX
        log_stds = jnp.clip(log_stds, log_std_min, log_std_max)

        if not self.tanh_squash_distribution:
            means = nn.tanh(means)

        base_dist = tfd.MultivariateNormalDiag(loc=means,
                                               scale_diag=jnp.exp(log_stds) *
                                               temperature)
        if self.tanh_squash_distribution:
            return tfd.TransformedDistribution(distribution=base_dist,
                                               bijector=tfb.Tanh())
        else:
            return base_dist


@functools.partial(jax.jit, static_argnames=('actor_def', 'distribution'))
def _sample_actions(rng: PRNGKey,
                    actor_def: nn.Module,
                    actor_params: Params,
                    observations: np.ndarray,
                    temperature: float = 1.0) -> Tuple[PRNGKey, jnp.ndarray]:
    dist = actor_def.apply({'params': actor_params}, observations, temperature)
    rng, key = jax.random.split(rng)
    return rng, dist.sample(seed=key)


def sample_actions(rng: PRNGKey,
                   actor_def: nn.Module,
                   actor_params: Params,
                   observations: np.ndarray,
                   temperature: float = 1.0) -> Tuple[PRNGKey, jnp.ndarray]:
    return _sample_actions(rng, actor_def, actor_params, observations,
                           temperature)
