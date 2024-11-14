# Proximal Policy Optimization

We provide the following four multi-agent extensions to [PPO](https://arxiv.org/pdf/1707.06347).
* [ff-IPPO](https://github.com/instadeepai/Mava/blob/feat/develop/mava/systems/ppo/anakin/ff_ippo.py)
* [ff-MAPPO](https://github.com/instadeepai/Mava/blob/feat/develop/mava/systems/ppo/anakin/ff_mappo.py)
* [rec-IPPO](https://github.com/instadeepai/Mava/blob/feat/develop/mava/systems/ppo/anakin/rec_ippo.py)
* [rec-MAPPO](https://github.com/instadeepai/Mava/blob/feat/develop/mava/systems/ppo/anakin/rec_mappo.py)

In all cases IPPO implies that it is an implementation following the independent learners MARL paradigm while MAPPO implies that the implementation follows the centralised training with decentralised execution paradigm by having a centralised critic during training. The `ff` and `rec` in the system names implies that the policies are MLPs or have a [GRU](https://arxiv.org/pdf/1406.1078) memory module to help learning despite partial observability in the environment.

## Relevant papers:
* [Single agent Proximal Policy Optimization Algorithms](https://arxiv.org/pdf/1707.06347)
* [The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games](https://arxiv.org/pdf/2103.01955)
