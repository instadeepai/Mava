# Proximal Policy Optimization

We provide 4 implementations of multi-agent PPO.
* [ff-IPPO](https://github.com/instadeepai/Mava/blob/feat/develop/mava/systems/ppo/anakin/ff_ippo.py): feed forward independant PPO
* [ff-MAPPO](https://github.com/instadeepai/Mava/blob/feat/develop/mava/systems/ppo/anakin/ff_mappo.py): feed forward multi-agent PPO
* [rec-IPPO](https://github.com/instadeepai/Mava/blob/feat/develop/mava/systems/ppo/anakin/rec_ippo.py): recurrent independant PPO
* [rec-MAPPO](https://github.com/instadeepai/Mava/blob/feat/develop/mava/systems/ppo/anakin/rec_mappo.py): recurrent multi-agent PPO

Where independant PPO uses independant learners and multi-agent PPO uses a CTDE style of training with a centralized critic.

## Relevant papers:
* [Single agent Proximal Policy Optimization Algorithms](https://arxiv.org/pdf/1707.06347)
* [The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games](https://arxiv.org/pdf/2103.01955)
