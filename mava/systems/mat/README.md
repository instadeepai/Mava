# Multi-agent Transformer

We provide an implementation of the Multi-agent Transformer algorithm in JAX. MAT casts cooperative multi-agent reinforcement learning as a sequence modelling problem where agent observations and actions are treated as a sequence. At each timestep the observations of all agents are encoded and then these encoded observations are used for auto-regressive action selection.

## Relevant paper:
* [Multi-Agent Reinforcement Learning is a Sequence Modeling Problem](https://arxiv.org/pdf/2205.14953)
