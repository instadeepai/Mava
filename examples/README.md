# Examples
We include a non-exhustive number of examples, showing common use-cases for Mava. We also have a [Quickstart notebook][quickstart] that can be used to quickly create and trian your first Multi-Agent System.

## Environments

In Mava we support a variety of different environments, which include
[PettingZoo][pettingzoo], [SMAC][smac], [2D RoboCup][robocup], [Flatland][flatland], [OpenSpiel][openspiel] environments, as well as a few custom [environments][debug] inside Mava.

## Continuous control
We include a number of systems running on continuous control tasks.

- Debugging Environment - Simple Spread
    -   **MA-DDPG**:
        an MA-DDPG system on the continuous action space simple_spread MPE debugging environment.
        - [(Feedforward, centralised)](debugging/simple_spread/feedforward/centralised/run_maddpg.py)
        - [(Feedforward, decentralised)](debugging/simple_spread/feedforward/decentralised/run_maddpg.py)
        - [(Feedforward, decentralised)](debugging/simple_spread/feedforward/decentralised/run_maddpg_record.py) - ***Example recording agents acting in the environment***.
        - [(Feedforward, networked)](debugging/simple_spread/feedforward/networked/run_maddpg.py)
        - [(Feedforward, networked)](debugging/simple_spread/feedforward/networked/run_maddpg_custom_network.py) - ***Example using a custom, sparse, networked architecture.***
        - [(Feedforward, state_based)](debugging/simple_spread/feedforward/state_based/run_maddpg.py)
        - [(Recurrent, decentralised)](debugging/simple_spread/recurrent/decentralised/run_maddpg.py)
        - [(Recurrent, state_based)](debugging/simple_spread/recurrent/state_based/run_maddpg.py)

    -   **MA-D4PG**:
        an MA-D4PG system on the continuous action space simple_spread MPE debugging environment.
        - [(Feedforward, centralised)](debugging/simple_spread/feedforward/centralised/run_mad4pg.py)
        - [(Feedforward, decentralised)](debugging/simple_spread/feedforward/decentralised/run_mad4pg.py)
        - [(Feedforward, state_based)](debugging/simple_spread/feedforward/state_based/run_mad4pg.py)
        - [(Recurrent, decentralised)](debugging/simple_spread/recurrent/decentralised/run_mad4pg.py)

- PettingZoo - Multiwalker
    -   **MA-DDPG**:
        an MA-DDPG system on the Multiwalker environment.
        - [(Feedforward, centralised)](petting_zoo/sisl/multiwalker/feedforward/centralised/run_maddpg.py)
        - [(Feedforward, decentralised)](petting_zoo/sisl/multiwalker/feedforward/decentralised/run_maddpg.py)
        - [(Recurrent, decentralised)](petting_zoo/sisl/multiwalker/recurrent/decentralised/run_maddpg.py)

    -   **MA-D4PG**:
        an MA-D4PG system on the Multiwalker environment.
        - [(Feedforward, decentralised)](petting_zoo/sisl/multiwalker/feedforward/decentralised/run_mad4pg.py)
        - [(Feedforward, decentralised)](petting_zoo/sisl/multiwalker/feedforward/decentralised/run_mad4pg.py) -***Example recording agents acting in the environment***.

- RoboCup
    -   **MA-D4PG**:
        an MA-D4PG system on the RoboCup environment.
        - [(Recurrent, state_based)](robocup/recurrent/state_based/run_mad4pg.py)
## Discrete control

We also include a number of systems running on discrete action space environments.

- Debugging Environment - Simple Spread
    -   **MA-PPO**:
        an MA-PPO system on the discrete action space simple_spread MPE debugging environment.
        - [(Feedforward, centralised)](debugging/simple_spread/feedforward/centralised/run_mappo.py)
        - [(Feedforward, decentralised)](debugging/simple_spread/feedforward/decentralised/run_mappo.py)

    -   **MA-DQN**:
        an MA-DQN system on the discrete action space simple_spread MPE debugging environment.
        - [(Feedforward, decentralised)](debugging/simple_spread/feedforward/decentralised/run_madqn.py)
        - [(Recurrent, decentralised)](debugging/simple_spread/recurrent/decentralised/run_madqn.py)
        - [(Recurrent, decentralised)](debugging/simple_spread/recurrent/decentralised/run_madqn.py) - ***Example using a system with communication.***

    -   **QMIX**:
        a QMIX system on the discrete action space simple_spread MPE debugging environment.
        - [(Feedforward, decentralised)](debugging/simple_spread/feedforward/decentralised/run_qmix.py)

    -   **VDN**:
        a VDN system on the discrete action space simple_spread MPE debugging environment.
        - [(Feedforward, decentralised)](debugging/simple_spread/feedforward/decentralised/run_vdn.py)

    -   **DIAL**:
        a DIAL system on the discrete action space simple_spread MPE debugging environment.
        - [(Recurrent, decentralised)](debugging/simple_spread/recurrent/decentralised/run_dial.py)

- Debugging Environment - Switch
    -    **DIAL**:
        a DIAL system on the discrete custom SwitchGame environment.
            - [(Recurrent, decentralised)](debugging/switch/recurrent/decentralised/run_dial.py)
- PettingZoo - Multi-Agent Atari
    -   **MA-DQN**:
        an MA-DQN system on two player competitive Atari Pong.
        - [(Feedforward, decentralised)](petting_zoo/atari/pong/feedforward/decentralised/run_madqn.py)

- PettingZoo - Multi-Agent Particle Environment
    -   **MA-DDPG**:
        an MA-DDPG system on Simple Speaker Listener.
        - [(Feedforward, decentralised)](petting_zoo/mpe/simple_speaker_listener/feedforward/decentralised/run_maddpg.py)

- PettingZoo - Multi-Agent Particle Environment
    -   **MA-DDPG**:
        an MA-DDPG system on Simple Spread.
        - [(Feedforward, decentralised)](petting_zoo/mpe/simple_spread/feedforward/decentralised/run_maddpg.py)

- SMAC - StarCraft Multi-Agent Challenge
    -   **MA-DQN**:
        an MA-DQN system on SMAC environment.
        - [(Feedforward, decentralised)](smac/feedforward/decentralised/run_madqn.py)
        - [(Recurrent, decentralised)](smac/recurrent/decentralised/run_madqn.py)

    -   **QMIX**:
        a QMIX system on SMAC environment.
        - [(Feedforward, decentralised)](smac/feedforward/decentralised/run_qmix.py)

    -   **VDN**:
        a VDN system on SMAC environment.
        - [(Feedforward, decentralised)](smac/feedforward/decentralised/run_vdn.py)

- OpenSpiel - Tic Tac Toe
    -   **MA-DQN**:
        an MA-DQN system on OpenSpiel environment.
        - [(Feedforward, decentralised)](openspiel/tic_tac_toe/feedforward/decentralised/run_madqn.py)





[debug]: ../mava/utils/debugging
[pettingzoo]: https://github.com/PettingZoo-Team/PettingZoo
[flatland]: https://gitlab.aicrowd.com/flatland/flatland
[smac]: https://github.com/oxwhirl/smac
[openspiel]: https://github.com/deepmind/open_spiel
[robocup]: https://github.com/rcsoccersim
[quickstart]: ./quickstart.ipynb
