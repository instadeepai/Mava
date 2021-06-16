# Examples
This directory includes working examples of Mava systems.

## Environments

In Mava we use a variety of different environments which include the
[PettingZoo][pettingzoo] and [Flatland][flatland] environment set as well as a few custom [environments][debug] inside Mava.

## Continuous control
We include a number of systems running on continuous control tasks.

-   [MA-DDPG (centralised, feedforward)](debugging_envs/run_centralised_feedforward_maddpg.py):
    an MA-DDPG system on the continuous action space simple_spread MPE debugging environment.
-   [PPO (centralised, feedforward)](debugging_envs/run_centralised_feedforward_mappo.py):
    a PPO system on the continuous action space simple_spread MPE debugging environment.
-   [MA-D4PG (decentralised, feedforward)](debugging_envs/run_decentralised_feedforward_mad4pg.py):
    an MA-D4PG system on the continuous action space simple_spread MPE debugging environment.
-   [MA-DDPG (decentralised, feedforward)](debugging_envs/run_decentralised_feedforward_maddpg.py):
    an MA-DDPG system on the continuous action space simple_spread MPE debugging environment.
-   [PPO (decentralised, feedforward)](debugging_envs/run_decentralised_feedforward_mappo.py):
    a PPO system on the continuous action space simple_spread MPE debugging environment.
-   [MA-D4PG (decentralised, recurrent)](debugging_envs/run_decentralised_recurrent_mad4pg.py):
    an MA-D4PG system on the continuous action space simple_spread MPE debugging environment.
-   [MA-DDPG (decentralised, recurrent)](debugging_envs/run_decentralised_recurrent_maddpg.py):
    an MA-DDPG system on the continuous action space simple_spread MPE debugging environment.
-   [MA-DDPG (decentralised, feedforward)](debugging_envs/run_feedforward_maddpg_record_video.py):
    an MA-DDPG system (with video wrapping) on the continuous action space simple_spread MPE debugging environment.
-   [MA-DDPG (networked, feedforward)](debugging_envs/run_networked_feedforward_maddpg.py):
    an MA-DDPG system on the continuous action space simple_spread MPE debugging environment.
-   [MA-DDPG (state-based, feedforward)](debugging_envs/run_state_based_feedforward_maddpg.py):
    an MA-DDPG system on the continuous action space simple_spread MPE debugging environment.
-   [MA-DDPG (state-based, recurrent)](debugging_envs/run_state_based_recurrent_maddpg.py):
    an MA-DDPG system on the continuous action space simple_spread MPE debugging environment.
-   [PPO (centralised, feedforward)](petting_zoo/run_centralised_feedforward_mappo.py):
    a PPO system using a centralised critic on the continuous action space multiwalker_v7 PettingZoo environment.
-   [MA-D4PG (decentralised, feedforward)](petting_zoo/run_decentralised_feedforward_mad4pg.py):
    an MA-D4PG system on the continuous action space multiwalker_v7 PettingZoo environment.
-   [MA-DDPG (decentralised, feedforward)](petting_zoo/run_decentralised_feedforward_maddpg.py):
    an MA-DDPG system on the continuous action space multiwalker_v7 PettingZoo environment.
-   [MA-DDPG (decentralised, recurrent)](petting_zoo/run_decentralised_recurrent_maddpg.py):
    an MA-DDPG system on the continuous action space multiwalker_v7 PettingZoo environment.
-   [MA-DDPG (decentralised, feedforward)](petting_zoo/run_feedforward_maddpg_record_video.py):
    an MA-DDPG system (with video wrapping) on the continuous action space multiwalker_v7 PettingZoo environment.
-   [MA-DDPG (networked, feedforward)](petting_zoo/run_networked_feedforward_maddpg_sparse.py):
    an MA-DDPG system oon the continuous action space multiwalker_v7 PettingZoo environm
[petting_zoo]: https://github.com/PettingZoo-Team/PettingZoo

## Discrete control

We also include a number of systems running on discrete action space environments.
-   [MA-D4PG (decentralised, feedforward)](debugging_envs/run_decentralised_feedforward_discrete_mad4pg.py):
    an MA-D4PG system on the discrete action space simple_spread MPE debugging environment.
-   [MA-DDPG (decentralised, feedforward)](debugging_envs/run_decentralised_feedforward_discrete_maddpg.py):
    an MA-DDPG system on the discrete action space simple_spread MPE debugging environment.
-   [MA-DQN (feedforward)](debugging_envs/run_feedforward_madqn.py):
    an MA-DQN system running on the discrete action space simple_spread MPE debugging environment.
-   [MA-DQN (feedforward)](debugging_envs/run_feedforward_madqn.py):
    an MA-DQN system (with fingerprints) running on the discrete action space simple_spread MPE debugging environment.
-   [QMIX](debugging_envs/run_feedforward_qmix.py):
    a QMIX system on the two step custom environment.
-   [VDN](debugging_envs/run_feedforward_vdn.py):
    a VDN system on the two step custom environment.
-   [DIAL](debugging_envs/run_recurrent_dial.py):
    a DIAL system on the custom SwitchGame environment.
-   [DIAL](debugging_envs/run_recurrent_dial_spread.py):
    a DIAL system on the discrete action space simple_spread MPE debugging environment.
-   [MA-DQN (recurrent)](debugging_envs/run_recurrent_madqn.py):
    an MA-DQN system running on the discrete action space simple_spread MPE debugging environment.
-   [MA-DQN (feedforward)](petting_zoo/run_feedforward_madqn.py):
    an MA-DQN system running on the continuous action space multiwalker_v7 PettingZoo environment.
-   [MA-DQN (feedforward)](petting_zoo/run_feedforward_madqn.py):
    an MA-DQN system (with fingerprints) running on the continuous action space multiwalker_v7 PettingZoo environment.
-   [QMIX](petting_zoo/run_feedforward_qmix.py):
    a QMIX system on the two step custom environment.
-   [VDN](petting_zoo/run_feedforward_vdn.py):
    a VDN system on the two step custom environment.
-   [MA-DQN (feedforward)](smac/run_feedforward_madqn.py):
    an MA-DQN system running on the StarCraft 3m environment.
-   [MA-DQN (recurrent)](smac/run_recurrent_madqn.py):
    an MA-DQN system running on the StarCraft 3m environment.

[debug]: ../mava/utils/debugging
[pettingzoo]: https://github.com/PettingZoo-Team/PettingZoo
[flatland]: https://gitlab.aicrowd.com/flatland/flatland
