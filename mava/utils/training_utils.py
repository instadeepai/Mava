from typing import Any, Dict


# Checkpoint the networks.
def checkpoint_networks(system_checkpointer: Dict) -> None:
    if system_checkpointer and len(system_checkpointer.keys()) > 0:
        for network_key in system_checkpointer.keys():
            checkpointer = system_checkpointer[network_key]
            checkpointer.save()


# Map critic and polic losses to dict, grouped by agent.
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


# Map critic and polic losses to dict, grouped by agent.
def map_losses_per_agent_q(q_losses: Dict) -> Dict:
    assert len(q_losses) > 0
    logged_losses: Dict[str, Dict[str, Any]] = {}
    agents = q_losses.keys()
    for agent in agents:
        logged_losses[agent] = {
            "q_value_loss": q_losses[agent],
        }

    return logged_losses


# Safely delete object from class.
def safe_del(object_class: Any, attrname: str) -> None:
    try:
        if hasattr(object_class, attrname):
            delattr(object_class, attrname)
    except AttributeError:
        pass
