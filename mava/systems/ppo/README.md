# Proximal Policy Optimization

We provide the following four multi-agent extensions to [PPO](https://arxiv.org/pdf/1707.06347) following the Anakin architecture.

* [ff-IPPO](../../systems/ppo/anakin/ff_ippo.py)
* [ff-MAPPO](../../systems/ppo/anakin/ff_mappo.py)
* [rec-IPPO](../../systems/ppo/anakin/rec_ippo.py)
* [rec-MAPPO](../../systems/ppo/anakin/rec_mappo.py)

In all cases IPPO implies that it is an implementation following the independent learners MARL paradigm while MAPPO implies that the implementation follows the centralised training with decentralised execution paradigm by having a centralised critic during training. The `ff` or `rec` suffixes in the system names implies that the policy networks are MLPs or have a [GRU](https://arxiv.org/pdf/1406.1078) memory module to help learning despite partial observability in the environment.

In addition to the Anakin-based implementations, we also include a Sebulba-based implementation of [ff-IPPO](../../systems/ppo/sebulba/ff_ippo.py) which can be used on environments that are not written in JAX and adhere to the Gymnasium API.

## Relevant papers:
* [Proximal Policy Optimization Algorithms](https://arxiv.org/pdf/1707.06347)
* [The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games](https://arxiv.org/pdf/2103.01955)
* [Is Independent Learning All You Need in the StarCraft Multi-Agent Challenge?](https://arxiv.org/pdf/2011.09533)
