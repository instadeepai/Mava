# Q Learning

We provide two Q-Learning based systems that follow the independent learners and centralised training with decentralised execution paradigms:

* [rec-IQL](../../systems/q_learning/anakin/rec_iql.py)
* [rec-QMIX](../../systems/q_learning/anakin/rec_qmix.py)

`rec-IQL` is a multi-agent version of DQN that uses double DQN and has a GRU memory module and `rec-QMIX` is an implementation of QMIX in JAX that uses monontic value function decomposition.

## Relevant papers:
* [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/pdf/1312.5602)
* [Multiagent Cooperation and Competition with Deep Reinforcement Learning](https://arxiv.org/pdf/1511.08779)
* [QMIX: Monotonic Value Function Factorisation for
Deep Multi-Agent Reinforcement Learning](https://arxiv.org/pdf/1803.11485)
