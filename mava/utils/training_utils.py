from typing import Any, Dict


# Checkpoint the networks.
def checkpoint_networks(system_checkpointer: Dict) -> None:
    assert (
        system_checkpointer is not None and len(system_checkpointer.keys()) > 0
    ), "Invalid System Checkpointer."
    for network_key in system_checkpointer.keys():
        checkpointer = system_checkpointer[network_key]
        checkpointer.save()


def map_losses_per_agent_ac(critic_losses: Dict, policy_losses: Dict) -> Dict:
    assert len(policy_losses) > 0 and (
        len(critic_losses) == len(policy_losses)
    ), "Invalid System Checkpointer."
    logged_losses: Dict[str, Dict[str, Any]] = {}
    agents = policy_losses.keys()
    for agent in agents:
        logged_losses[agent] = {
            "critic_loss": critic_losses[agent],
            "policy_loss": policy_losses[agent],
        }

    return logged_losses
