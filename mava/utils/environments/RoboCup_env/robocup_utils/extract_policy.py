# type: ignore
import os
import pickle

from acme.tf import savers as tf2_savers


def save_policy_and_obs(architecture, agent_type, checkpoint_subpath, pickle_save_loc):
    policy_networks = architecture.create_actor_variables()
    before_sum = policy_networks["policies"][agent_type].variables[1].numpy().sum()
    # print("Weights before: ", before_sum)
    objects_to_save = {
        "policy": policy_networks["policies"][agent_type],
        "observation": policy_networks["observations"][agent_type],
    }

    checkpointer_dir = os.path.join(checkpoint_subpath, agent_type)
    tf2_savers.Checkpointer(
        time_delta_minutes=1,
        add_uid=False,
        directory=checkpointer_dir,
        objects_to_save=objects_to_save,
        enable_checkpointing=True,
    )
    after_sum = policy_networks["policies"][agent_type].variables[1].numpy().sum()

    assert before_sum != after_sum

    # Save policy variables
    print(policy_networks["policies"][agent_type])
    exit()
    policy = policy_networks["policies"][agent_type].set.variables
    policy_file = open(os.path.join(pickle_save_loc, "policy.obj"), "wb")
    pickle.dump(policy, policy_file)
    policy_file.close()

    # So observation variables
    observation = policy_networks["observations"][agent_type].variables
    obs_file = open(os.path.join(pickle_save_loc, "observations.obj"), "wb")
    pickle.dump(observation, obs_file)
    obs_file.close()


def load_policy_and_obs(pickle_save_loc):
    # Load policy variables
    filehandler = open(os.path.join(pickle_save_loc, "policy.obj"), "rb")
    policy_net = pickle.load(filehandler)

    # Load observation variables
    filehandler = open(os.path.join(pickle_save_loc, "policy.obj"), "rb")
    obs_net = pickle.load(filehandler)

    return policy_net, obs_net
