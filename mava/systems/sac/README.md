# Soft Actor-Critic

We provide the following three multi-agent extensions to the Soft Actor-Critic (SAC) algorithm.

* [ff-ISAC](../../systems/sac/anakin/ff_isac.py)
* [ff-MASAC](../../systems/sac/anakin/ff_masac.py)
* [ff-HASAC](../../systems/sac/anakin/ff_hasac.py)

`ISAC` is an implementation following the independent learners MARL paradigm while `MASAC` is an implementation that follows the centralised training with decentralised execution paradigm by having a centralised critic during training. `HASAC` follows the heterogeneous agent learning paradigm through sequential policy updates. The `ff` prefix to the algorithm names indicate that the algorithms use MLP-based policy networks.

## Relevant papers
* [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/pdf/1801.01290)
* [Multi-Agent Actor-Critic for Mixed
Cooperative-Competitive Environments](https://arxiv.org/pdf/1706.02275)
* [Robust Multi-Agent Control via Maximum Entropy
Heterogeneous-Agent Reinforcement Learning](https://arxiv.org/pdf/2306.10715)
