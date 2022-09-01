from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Type

import acme
import dm_env
import haiku as hk
import jax
import jax.numpy as jnp
import jraph
from acme.jax import utils
from jraph import GraphConvolution, GraphsTuple

from mava import specs as mava_specs
from mava.callbacks import Callback
from mava.components.jax import Component
from mava.components.jax.building import EnvironmentSpec
from mava.components.jax.building.networks import Networks
from mava.core_jax import SystemBuilder


def make_default_gcn(
    environment_spec: mava_specs.MAEnvironmentSpec,
    rng_key: List[int],
    update_node_layer_sizes: Sequence[Sequence[int]] = ((128, 128), (128, 128)),
) -> Dict[str, Any]:
    """Create default GCN to use for communication."""
    # Create agent_type specs.
    specs: Dict[
        str, acme.specs.EnvironmentSpec
    ] = environment_spec.get_agent_environment_specs()

    # Assumes all agents have same spec
    # TODO: assert all have same spec (i.e. same obs size)
    agent_spec: acme.specs.EnvironmentSpec = list(specs.values())[0]

    # Define GNN function
    def net_fn(graph: GraphsTuple) -> GraphsTuple:
        for sizes in update_node_layer_sizes:
            # One GNN layer per 'sizes'
            update_node_fn_seq = []
            for size in sizes:
                # One linear layer in node update per 'size'
                update_node_fn_seq.append(hk.Linear(size))
                update_node_fn_seq.append(jax.nn.relu)
            gnn = GraphConvolution(
                update_node_fn=hk.Sequential(update_node_fn_seq),
                add_self_edges=True,
            )
            graph = gnn(graph)

        # One final layer to get the output size equal to
        # the policy input size (i.e. env obs size)
        # TODO: rather init the policy on the output of the GNN
        obs_size = agent_spec.observations.observation.shape[0]
        graph_decoder = jraph.GraphMapFeatures(embed_node_fn=hk.Linear(obs_size))
        graph = graph_decoder(graph)

        return graph

    # Transform into pure function
    net_fn = hk.without_apply_rng(hk.transform(net_fn))

    # Init the network with dummy obs, dummy_graph, and a key
    dummy_obs = utils.zeros_like(agent_spec.observations.observation)
    num_agents = len(specs)
    dummy_graph = jraph.GraphsTuple(
        nodes=jnp.array([dummy_obs for _ in range(num_agents)]),
        edges=None,
        senders=jnp.array([]),
        receivers=jnp.array([]),
        n_node=jnp.array([num_agents]),
        n_edge=jnp.array([]),
        globals=None,
    )
    network_key, rng_key = jax.random.split(rng_key)
    params = net_fn.init(network_key, dummy_graph)
    print(params)  # TODO: save params to class with forward_fn

    return {"gdn_network": net_fn}


@dataclass
class GdnNetworksConfig:
    network_factory: Optional[Callable[[str], dm_env.Environment]] = None


class DefaultGdnNetworks(Component):
    def __init__(
        self,
        config: GdnNetworksConfig = GdnNetworksConfig(),
    ):
        """Component defines the default way to initialise Gdn networks.

        Args:
            config: GdnNetworksConfig.
        """
        self.config = config

    def on_building_init_start(self, builder: SystemBuilder) -> None:
        """Create and store the GDN network factory from the config.

        Args:
            builder: SystemBuilder.

        Returns:
            None.
        """
        # Build network function here
        network_key, builder.store.key = jax.random.split(builder.store.key)
        builder.store.gdn_network_factory = lambda: self.config.network_factory(
            environment_spec=builder.store.ma_environment_spec,
            agent_net_keys=builder.store.agent_net_keys,
            rng_key=network_key,
        )

    def on_building_init_end(self, builder: SystemBuilder) -> None:
        """Create the GDN networks from the factory.

        Args:
            builder: SystemBuilder.

        Returns:
            None.
        """
        builder.store.gdn_networks = builder.store.gdn_network_factory()

    @staticmethod
    def config_class() -> Optional[Callable]:
        """Config class used for component.

        Returns:
            config class/dataclass for component.
        """
        return GdnNetworksConfig

    @staticmethod
    def required_components() -> List[Type[Callback]]:
        """List of other Components required in the system for this Component to function.

        EnvironmentSpec required to set up builder.store.environment_spec.
        Networks required for config 'seed'.

        Returns:
            List of required component classes.
        """
        return [EnvironmentSpec, Networks]

    @staticmethod
    def name() -> str:
        """Static method that returns component name."""
        return "gdn_networks"
