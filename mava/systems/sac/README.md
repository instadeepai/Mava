# Soft Actor Critic

We provide 3 implementations of multi-agent SAC.
* [ff-ISAC](https://github.com/instadeepai/Mava/blob/feat/develop/mava/systems/sac/anakin/ff_isac.py): feed forward independant SAC
* [ff-MASAC](https://github.com/instadeepai/Mava/blob/feat/develop/mava/systems/sac/anakin/ff_masac.py): feed forward multi-agent SAC
* [ff-HASAC](https://github.com/instadeepai/Mava/blob/feat/develop/mava/systems/sac/anakin/ff_hasac.py): recurrent independant SAC

Where independant SAC uses independant learners and multi-agent SAC uses a CTDE style of training with a centralized critic and HASAC uses heterogenous style, sequential updates.

## Relevant papers
* [Single agent Soft Actor Critic](https://arxiv.org/pdf/1801.01290)
* [MADDPG](https://arxiv.org/pdf/1706.02275)
* [HASAC](https://arxiv.org/pdf/2306.10715)
