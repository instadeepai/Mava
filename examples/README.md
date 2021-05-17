# Examples
This directory includes working examples of Mava systems.

## Environments

In Mava we use a variety of different environments which include the
[petting_zoo] environment set as well as a few custom [environments](utils/debugging) inside Mava.

## Continuous control
We include a number of systems running on continuous control tasks.

-   [PPO (centralised)](debugging_envs/run_centralised_feedforward_mappo.py):
    a PPO system using a centralised critic on the continuous action space simple_spread MPE debugging environment.
-   [MA-DDPG (feedforward)](debugging_envs/run_feedforward_maddpg.py):
    an MA-DDPG system (without recurrence) on the continuous action space simple_spread MPE debugging environment.
-   [MA-D4PG](debugging_envs/run_feedforward_mad4pg.py):
    an MA-D4PG system on the continuous action space simple_spread MPE debugging environment.
-   [MA-DDPG (recurrent)](debugging_envs/run_recurrent_maddpg.py):
    an MA-DDPG system (with recurrence) on the continuous action space simple_spread MPE debugging environment.
-  [PPO (centralised)](petting_zoo/run_centralised_feedforward_mappo.py):
    a PPO system using a centralised critic on the continuous action space multiwalker_v7 PettingZoo environment.
-  [MA-DDPG (centralised)](petting_zoo/run_feedforward_maddpg.py):
    an MA-DDPG system on the continuous action space multiwalker_v7 PettingZoo environment.
- [MA-DDPG (recurrent)](petting_zoo/run_recurrent_maddpg.py):
    an MA-DDPG system (with recurrence) on the continuous action space multiwalker_v7 PettingZoo environment.

[petting_zoo]: https://github.com/PettingZoo-Team/PettingZoo

## Discrete control

We also include a number of systems running on discrete action space environments.

- [MA-DQN (feedforward)](debugging_envs/run_feedforward_madqn.py):
    an MA-DQN system running on the discrete action space simple_spread MPE debugging environment.
- [PPO](debugging_envs/run_feedforward_mappo.py):
    a PPO system using a decentralised critic on the discrete action space simple_spread MPE debugging environment.
- [QMIX](debugging_envs/run_feedforward_qmix.py):
    a QMIX system on the two step custom environment.
- [DIAL](debugging_envs/run_recurrent_dial.py):
    a DIAL system on the custom SwitchGame environment.
- [DIAL](debugging_envs/run_recurrent_dial_spread.py):
    a DIAL system on the discrete action space simple_spread MPE debugging environment.
- [MA-DQN (feedforward)](debugging_envs/run_feedforward_madqn.py):
    an MA-DQN system running on the discrete action space maze_craze_v2 PettingZoo environment.
