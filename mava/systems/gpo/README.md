# Multi-Agent Guided Policy Optimization

We provide an implementation of Multi-Agent Guided Policy Optimization (MAGPO) in JAX. MAGPO co-trains a centralized guider policy with decentralized learner policies, aligning them through constraints.
The guider policy enables coordinated exploration and leverages global information, while the learner policies ensure deployability in a decentralized manner under partial observability.
The centralized guider policy implementation is based on [Sable](https://github.com/instadeepai/Mava/tree/develop/mava/systems/sable).

## Relevant paper:
- [Multi-Agent Guided Policy Optimization](https://arxiv.org/pdf/2507.18059)
