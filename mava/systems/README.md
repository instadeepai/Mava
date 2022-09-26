# Systems

Mava includes a number of pre-built systems as listed below. All the systems are
implemented using [Launchpad](https://github.com/deepmind/launchpad), which is used for distributed
training. This allows for easy scaling of computational resources by changing only one variable.

Below we list the different systems in Mava based on the action spaces they use. More systems will be added in future Mava updates.

For our TF2-based systems (maddpg, madqn, vdn and qmix), please install [`v0.1.3`](https://github.com/instadeepai/Mava/releases/tag/0.1.3) of Mava (e.g. `pip install id-mava==0.1.3`). We will no longer be supporting these systems as we have moved to JAX-based systems.

### Mixed
We also have a system that works with either discrete or continuous action-spaces:

System                                                    | Paper                    | Code
-------------------------------------------------------- | :----------------------: | :--:
Multi-Agent Proximal Policy Optimization (MAPPO)        | [Yu et al., 2021], [Schroeder et al., 2020]      | [![Jax][Jax Logo]][IPPO_Jax]

<!-- Jax agents -->
[IPPO_Jax]: https://github.com/instadeepai/Mava/tree/develop/mava/systems/ippo

<!-- Papers -->
[Lowe et al., 2017]: https://arxiv.org/abs/1706.02275
[Barth-Maron et al., 2018]: https://arxiv.org/abs/1804.08617
[Sunehag et al., 2017]:  https://arxiv.org/abs/1706.05296
[Rashid et al., 2018]: https://arxiv.org/abs/1803.11485

[Horgan et al., 2018]: https://arxiv.org/abs/1803.00933
[Foerster et al., 2016]: https://arxiv.org/abs/1605.06676

[Yu et al., 2021]: https://arxiv.org/abs/2103.01955
[Schroeder et al., 2020]: https://arxiv.org/abs/2011.09533

[Jax Logo]: https://raw.githubusercontent.com/instadeepai/Mava/develop/docs/images/jax_logo_small.png
