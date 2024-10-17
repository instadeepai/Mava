from typing import Optional, Tuple, Union

import chex
import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax.distributions as tfd
from flax import linen as nn
from flax.linen.initializers import orthogonal

from mava.networks.attention import SelfAttention
from mava.networks.distributions import IdentityTransformation
import tensorflow_probability.substrates.jax.distributions as tfd

class SwiGLU(nn.Module):
    ffn_dim: int
    embed_dim: int

    def setup(self) -> None:
        self.W_1 = self.param(
            "W_1", nn.initializers.zeros, (self.embed_dim, self.ffn_dim)
        )
        self.W_G = self.param(
            "W_G", nn.initializers.zeros, (self.embed_dim, self.ffn_dim)
        )
        self.W_2 = self.param(
            "W_2", nn.initializers.zeros, (self.ffn_dim, self.embed_dim)
        )

    def __call__(self, x):
        return (jax.nn.swish(x @ self.W_G) * (x @ self.W_1)) @ self.W_2


class EncodeBlock(nn.Module):
    n_embd: int
    n_head: int
    n_agent: int
    masked: bool = False
    use_swiglu: bool = False
    use_rmsnorm: bool = False

    def setup(self) -> None:
        ln = nn.RMSNorm if self.use_rmsnorm else nn.LayerNorm
        self.ln1 = ln()
        self.ln2 = ln()

        self.attn = SelfAttention(self.n_embd, self.n_head, self.n_agent, self.masked)

        if self.use_swiglu:
            self.mlp = SwiGLU(self.n_embd, self.n_embd)
        else:
            self.mlp = nn.Sequential(
                [
                    nn.Dense(self.n_embd, kernel_init=orthogonal(jnp.sqrt(2))),
                    nn.gelu,
                    nn.Dense(self.n_embd, kernel_init=orthogonal(0.01)),
                ],
            )

    def __call__(self, x: chex.Array) -> chex.Array:
        x = self.ln1(x + self.attn(x, x, x))
        x = self.ln2(x + self.mlp(x))
        return x


class Encoder(nn.Module):
    obs_dim: int
    action_dim: int
    n_block: int
    n_embd: int
    n_head: int
    n_agent: int
    use_swiglu: bool = False
    use_rmsnorm: bool = False

    def setup(self) -> None:
        ln = nn.RMSNorm if self.use_rmsnorm else nn.LayerNorm

        self.obs_encoder = nn.Sequential(
            [ln(), nn.Dense(self.n_embd, kernel_init=orthogonal(jnp.sqrt(2))), nn.gelu],
        )
        self.ln = ln()
        self.blocks = nn.Sequential(
            [
                EncodeBlock(
                    self.n_embd,
                    self.n_head,
                    self.n_agent,
                    use_swiglu=self.use_swiglu,
                    use_rmsnorm=self.use_swiglu,
                )
                for _ in range(self.n_block)
            ]
        )
        self.head = nn.Sequential(
            [
                nn.Dense(self.n_embd, kernel_init=orthogonal(jnp.sqrt(2))),
                nn.gelu,
                ln(),
                nn.Dense(1, kernel_init=orthogonal(0.01)),
            ],
        )

    def __call__(self, obs: chex.Array) -> Tuple[chex.Array, chex.Array]:
        obs_embeddings = self.obs_encoder(obs)
        x = obs_embeddings

        rep = self.blocks(self.ln(x))
        v_loc = self.head(rep)

        return jnp.squeeze(v_loc, axis=-1), rep


class DecodeBlock(nn.Module):
    n_embd: int
    n_head: int
    n_agent: int
    masked: bool = True
    use_swiglu: bool = False
    use_rmsnorm: bool = False

    def setup(self) -> None:
        ln = nn.RMSNorm if self.use_rmsnorm else nn.LayerNorm
        self.ln1 = ln()
        self.ln2 = ln()
        self.ln3 = ln()

        self.attn1 = SelfAttention(self.n_embd, self.n_head, self.n_agent, self.masked)
        self.attn2 = SelfAttention(self.n_embd, self.n_head, self.n_agent, self.masked)

        if self.use_swiglu:
            self.mlp = SwiGLU(self.n_embd, self.n_embd)
        else:
            self.mlp = nn.Sequential(
                [
                    nn.Dense(self.n_embd, kernel_init=orthogonal(jnp.sqrt(2))),
                    nn.gelu,
                    nn.Dense(self.n_embd, kernel_init=orthogonal(0.01)),
                ],
            )

    def __call__(self, x: chex.Array, rep_enc: chex.Array) -> chex.Array:
        x = self.ln1(x + self.attn1(x, x, x))
        x = self.ln2(rep_enc + self.attn2(key=x, value=x, query=rep_enc))
        x = self.ln3(x + self.mlp(x))
        return x


class Decoder(nn.Module):
    obs_dim: int
    action_dim: int
    n_block: int
    n_embd: int
    n_head: int
    n_agent: int
    action_space_type: str = "discrete"
    use_swiglu: bool = False
    use_rmsnorm: bool = False

    def setup(self) -> None:
        ln = nn.RMSNorm if self.use_rmsnorm else nn.LayerNorm

        if self.action_space_type == "discrete":
            self.action_encoder = nn.Sequential(
                [
                    nn.Dense(
                        self.n_embd, use_bias=False, kernel_init=orthogonal(jnp.sqrt(2))
                    ),
                    nn.gelu,
                ],
            )
        else:
            self.action_encoder = nn.Sequential(
                [nn.Dense(self.n_embd, kernel_init=orthogonal(jnp.sqrt(2))), nn.gelu],
            )
            self.log_std = self.param(
                "log_std", nn.initializers.zeros, (self.action_dim,)
            )

        # Always initialize log_std but set to None for discrete action spaces
        # This ensures the attribute exists but signals it should not be used.
        if self.action_space_type == "discrete":
            self.log_std = None

        self.obs_encoder = nn.Sequential(
            [ln(), nn.Dense(self.n_embd, kernel_init=orthogonal(jnp.sqrt(2))), nn.gelu],
        )
        self.ln = ln()
        self.blocks = [
            DecodeBlock(
                self.n_embd,
                self.n_head,
                self.n_agent,
                use_swiglu=self.use_swiglu,
                use_rmsnorm=self.use_swiglu,
                name=f"cross_attention_block_{block_id}",
            )
            for block_id in range(self.n_block)
        ]
        self.head = nn.Sequential(
            [
                nn.Dense(self.n_embd, kernel_init=orthogonal(jnp.sqrt(2))),
                nn.gelu,
                ln(),
                nn.Dense(self.action_dim, kernel_init=orthogonal(0.01)),
            ],
        )

    def __call__(
        self, action: chex.Array, obs_rep: chex.Array
    ) -> chex.Array:
        action_embeddings = self.action_encoder(action)
        x = self.ln(action_embeddings)

        # Need to loop here because the input and output of the blocks are different.
        # Blocks take an action embedding and observation encoding as input but only give the cross
        # attention output as output.
        for block in self.blocks:
            x = block(x, obs_rep)
        logit = self.head(x)

        return logit


class MultiAgentTransformer(nn.Module):
    obs_dim: int
    action_dim: int
    n_block: int
    n_embd: int
    n_head: int
    n_agent: int
    action_space_type: str = "discrete"
    use_swiglu: bool = False
    use_rmsnorm: bool = False

    def setup(self) -> None:
        if self.action_space_type not in ["discrete", "continuous"]:
            raise ValueError(f"Invalid action space type: {self.action_space_type}")

        self.encoder = Encoder(
            self.obs_dim,
            self.action_dim,
            self.n_block,
            self.n_embd,
            self.n_head,
            self.n_agent,
            use_swiglu=self.use_swiglu,
            use_rmsnorm=self.use_rmsnorm,
        )
        self.decoder = Decoder(
            self.obs_dim,
            self.action_dim,
            self.n_block,
            self.n_embd,
            self.n_head,
            self.n_agent,
            self.action_space_type,
            use_swiglu=self.use_swiglu,
            use_rmsnorm=self.use_rmsnorm,
        )
        if self.action_space_type == "discrete":
            self.autoregressive_act = discrete_autoregressive_act
            self.parallel_act = discrete_parallel_act

        else:
            self.autoregressive_act = continuous_autoregressive_act
            self.parallel_act = continuous_parallel_act

    def __call__(
        self, obs: chex.Array, action: chex.Array, legal_actions: chex.Array, key: chex.PRNGKey
    ) -> Tuple[chex.Array, chex.Array, chex.Array]:
        v_loc, obs_rep = self.encoder(obs)

        action_log, entropy = self.parallel_act(
            decoder=self.decoder,
            obs_rep=obs_rep,
            batch_size=obs.shape[0],
            action=action,
            n_agent=self.n_agent,
            action_dim=self.action_dim,
            legal_actions=legal_actions,
            key=key,
        )

        return action_log, v_loc, entropy

    def get_actions(
        self, obs: chex.Array, legal_actions: chex.Array, key: chex.PRNGKey
    ) -> Tuple[chex.Array, chex.Array, chex.Array, Optional[chex.Array]]:
        # obs: (batch, n_agent, obs_dim)
        # obs_rep: (batch, n_agent, n_embd)
        # v_loc: (batch, n_agent, 1)

        v_loc, obs_rep = self.encoder(obs)
        output_action, output_action_log = self.autoregressive_act(
            decoder=self.decoder,
            obs_rep=obs_rep,
            batch_size=obs.shape[0],
            n_agent=self.n_agent,
            action_dim=self.action_dim,
            legal_actions=legal_actions,
            key=key,
        )
        return output_action, output_action_log, v_loc


def discrete_parallel_act(
    decoder: Decoder,
    obs_rep: chex.Array,  # (batch, n_agent, n_embd)
    action: chex.Array,  # (batch, n_agent, 1)
    batch_size: int,  # (, )
    n_agent: int,  # (, )
    action_dim: int,  # (, )
    legal_actions: chex.Array,
    key: chex.PRNGKey,
) -> Tuple[chex.Array, chex.Array]:
    one_hot_action = jax.nn.one_hot(action, action_dim)  # (batch, n_agent, action_dim)
    shifted_action = jnp.zeros(
        (batch_size, n_agent, action_dim + 1)
    )  # (batch, n_agent, action_dim + 1)
    shifted_action = shifted_action.at[:, 0, 0].set(1)
    # This should look like this for all batches:
    # [[1, 0, 0, 0, 0, 0],
    #  [0, 0, 0, 0, 0, 0],
    #  [0, 0, 0, 0, 0, 0]]
    shifted_action = shifted_action.at[:, 1:, 1:].set(one_hot_action[:, :-1, :])
    # If the action is:
    # [[2],
    #  [1],
    #  [0]]

    # The one hot action is:
    # [[0, 0, 1, 0, 0],
    #  [0, 1, 0, 0, 0],
    #  [1, 0, 0, 0, 0]]

    # The shifted action will be:
    # [[1, 0, 0, 0, 0, 0],
    #  [0, 0, 0, 1, 0, 0],
    #  [0, 0, 1, 0, 0, 0]]

    logit = decoder(shifted_action, obs_rep)  # (batch, n_agent, action_dim)

    masked_logits = jnp.where(
        legal_actions,
        logit,
        jnp.finfo(jnp.float32).min,
    )

    distribution = IdentityTransformation(distribution=tfd.Categorical(logits=masked_logits))
    action_log_prob = distribution.log_prob(action)
    entropy = distribution.entropy(seed=key)

    return action_log_prob, entropy


def continuous_parallel_act(
    decoder: Decoder,
    obs_rep: chex.Array,  # (batch, n_agent, n_embd)
    obs: chex.Array,  # (batch, n_agent, obs_dim)
    action: chex.Array,  # (batch, n_agent, 1 <- should prob be action_dim)
    batch_size: int,  # (, )
    n_agent: int,  # (, )
    action_dim: int,  # (, )
    legal_actions: chex.Array,
) -> Tuple[chex.Array, chex.Array]:
    shifted_action = jnp.zeros(
        (batch_size, n_agent, action_dim)
    )  # (batch, n_agent, action_dim)

    shifted_action = shifted_action.at[:, 1:, :].set(action[:, :-1, :])

    act_mean = decoder(shifted_action, obs_rep, obs)
    action_std = jax.nn.softplus(decoder.log_std)

    distribution = tfd.MultivariateNormalDiag(loc=act_mean, scale_diag=action_std)
    action_log_prob = distribution.log_prob(action)
    action_log_prob -= jnp.sum(
        2.0 * (jnp.log(2.0) - action - jax.nn.softplus(-2.0 * action)), axis=-1
    )  # (batch, n_agent, 1)
    entropy = distribution.entropy()

    return action_log_prob, entropy


def discrete_autoregressive_act(
    decoder: Decoder,
    obs_rep: chex.Array,
    batch_size: int,
    n_agent: int,
    action_dim: int,
    legal_actions: chex.Array,
    key: chex.PRNGKey,
) -> Tuple[chex.Array, chex.Array, None]:
    shifted_action = jnp.zeros((batch_size, n_agent, action_dim + 1))  
    # (batch, n_agent, action_dim + 1)
    shifted_action = shifted_action.at[:, 0, 0].set(1)
    # This should look like:
    # [[1, 0, 0, 0, 0, 0],
    #  [0, 0, 0, 0, 0, 0],
    #  [0, 0, 0, 0, 0, 0]]
    output_action = jnp.zeros((batch_size, n_agent))
    output_action_log = jnp.zeros_like(output_action)
    # both have shape (batch, n_agent)

    for i in range(n_agent):
        logit = decoder(shifted_action, obs_rep)[:, i, :]
        # logit: (batch, action_dim)
        masked_logits = jnp.where(
            legal_actions[:, i, :],
            logit,
            jnp.finfo(jnp.float32).min,
        )
        key, sample_key = jax.random.split(key)

        distribution = IdentityTransformation(distribution=tfd.Categorical(logits=masked_logits))
        action = distribution.sample(seed=sample_key)
        action_log = distribution.log_prob(action)

        output_action = output_action.at[:, i].set(
            action
        )  # (batch, n_agent)
        output_action_log = output_action_log.at[:, i].set(
            action_log
        )  # (batch, n_agent)

        update_shifted_action = i + 1 < n_agent
        shifted_action = jax.lax.cond(
            update_shifted_action,
            lambda action=action, i=i, shifted_action=shifted_action: shifted_action.at[
                :, i + 1, 1:
            ].set(jax.nn.one_hot(action, action_dim)),
            lambda shifted_action=shifted_action: shifted_action,
        )

        # An example of a shifted action:
        # [[1, 0, 0, 0, 0, 0],
        #  [0, 0, 0, 0, 0, 1],
        #  [0, 0, 0, 1, 0, 0]]

        # Assuming the actions where [4, 2, 4]
        # An important note, the shifted actions are not really relevant,
        # they are just used to act autoregreesively.

    return output_action.astype(jnp.int32), output_action_log


def continuous_autoregressive_act(
    decoder: Decoder,
    obs_rep: chex.Array,  # (batch, n_agent, n_embd)
    obs: chex.Array,  # (batch, n_agent, obs_dim)
    batch_size: int,  # (, )
    n_agent: int,  # (, )
    action_dim: int,  # (, )
    legal_actions: Union[chex.Array, None],
    key: chex.PRNGKey,
) -> Tuple[chex.Array, chex.Array, chex.Array]:
    shifted_action = jnp.zeros(
        (batch_size, n_agent, action_dim)
    )  # (batch, n_agent, action_dim)
    output_action = jnp.zeros((batch_size, n_agent, action_dim))
    raw_output_action = jnp.zeros((batch_size, n_agent, action_dim))
    output_action_log = jnp.zeros((batch_size, n_agent))

    for i in range(n_agent):
        act_mean = decoder(shifted_action, obs_rep, obs)[:, i, :]  # (batch, action_dim)
        action_std = action_std = jax.nn.softplus(decoder.log_std)

        key, sample_key = jax.random.split(key)

        distribution = tfd.MultivariateNormalDiag(loc=act_mean, scale_diag=action_std)
        raw_action = distribution.sample(seed=sample_key)
        action_log = distribution.log_prob(raw_action)
        action_log -= jnp.sum(
            2.0 * (jnp.log(2.0) - raw_action - jax.nn.softplus(-2.0 * raw_action)),
            axis=-1,
        )  # (batch, 1)
        action = jnp.tanh(raw_action)

        raw_output_action = raw_output_action.at[:, i, :].set(raw_action)
        output_action = output_action.at[:, i, :].set(action)
        output_action_log = output_action_log.at[:, i].set(action_log)

        update_shifted_action = i + 1 < n_agent
        shifted_action = jax.lax.cond(
            update_shifted_action,
            lambda action=action, i=i, shifted_action=shifted_action: shifted_action.at[
                :, i + 1, :
            ].set(action),
            lambda shifted_action=shifted_action: shifted_action,
        )

    return output_action, output_action_log, raw_output_action


if __name__ == "__main__":
    obs_dim = 21
    action_dim = 5
    n_block = 2
    n_embed = 12
    n_head = 6
    n_agent = 3
    mat = MultiAgentTransformer(obs_dim, action_dim, n_block, n_embed, n_head, n_agent)
    mock_obs = jnp.zeros((1, n_agent, obs_dim))
    mock_action = jnp.array([[4, 2, 4]])
    mock_legal_action = jnp.ones((1, n_agent, action_dim))
    params = mat.init(jax.random.PRNGKey(0), mock_obs, mock_action, mock_legal_action)
    output_action, output_action_log_prob, v_loc = mat.apply(
        params,
        mock_obs,
        mock_legal_action,
        jax.random.PRNGKey(42),
        method=mat.get_actions,
    )
    print(output_action.shape, output_action_log_prob.shape, v_loc.shape)
    value = mat.apply(params, mock_obs, method=mat.get_values)
    print(value.shape)