# Systems

As with Acme, Mava includes a number of pre-built agents as listed below. All the systems are
implemented using [Launchpad](https://github.com/deepmind/launchpad), which is used for distributed
training. This allows for easy scaling of computational resources by changing only one variable.

Below we list the different systems in Mava based on the action spaces they use. More systems will be added in future Mava updates.

### Continuous control

The following systems focus on this
setting:

Agent                                                                | Paper                    | Code
-------------------------------------------------------------------- | :----------------------: | :--:
Multi-Agent Deep Deterministic Policy Gradient (MA-DDPG)             | [Lowe et al., 2017]   | [![TF](../../docs/images/tf-small.png)][MADDPG_TF2]
Multi-Agent Distributed Distributional DDPG (MA-D4PG)    | [Barth-Maron et al., 2018] | [![TF](../../docs/images/tf-small.png)][MAD4PG_TF2]

### Discrete control
We also include a number of systems built with discrete action-spaces in mind listed below:

Agent                                                    | Paper                    | Code
-------------------------------------------------------- | :----------------------: | :--:
Deep Q-Networks (DQN)                                    | [Horgan et al., 2018]      | [![TF](../../docs/images/tf-small.png)][DQN_TF2]
Differentiable Inter-Agent Learning (DIAL)               | [Foerster et al., 2016]    | [![TF](../../docs/images/tf-small.png)][DIAL_TF2]
QMIX                                                     | [Rashid et al., 2018]      | [![TF](../../docs/images/tf-small.png)][QMIX_TF2]

### Mixed
We also have a system that works with either discrete or continuous action-spaces:

Agent                                                    | Paper                    | Code
-------------------------------------------------------- | :----------------------: | :--:
Multi-Agent Proximal Policy Optimization (MA-PPO)        | [Yu et al., 2021], [Schroeder et al., 2020]      | [![TF](../../docs/images/tf-small.png)][MAPPO_TF2]

<!-- TF agents -->

[MADDPG_TF2]: tf/maddpg/
[MAD4PG_TF2]: tf/mad4pg/

[DQN_TF2]: tf/madqn/
[DIAL_TF2]: tf/dial/
[QMIX_TF2]: tf/qmix/

[MAPPO_TF2]: tf/mappo/

<!-- Papers -->
[Lowe et al., 2017]: https://arxiv.org/abs/1706.02275
[Barth-Maron et al., 2018]: https://arxiv.org/abs/1804.08617
[Rashid et al., 2018]: https://arxiv.org/abs/1803.11485

[Horgan et al., 2018]: https://arxiv.org/abs/1803.00933
[Foerster et al., 2016]: https://arxiv.org/abs/1605.06676

[Yu et al., 2021]: https://arxiv.org/abs/2103.01955
[Schroeder et al., 2020]: https://arxiv.org/abs/2011.09533
