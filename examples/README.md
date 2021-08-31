# Examples
We include a non-exhaustive number of examples, showing common use-cases for Mava. We also have a [Quickstart notebook][quickstart] that can be used to quickly create and train your first Multi-Agent System.

## Environments

In Mava, we support a variety of different environments, which include
PettingZoo ([Repo][pz_repo], [Paper][pz_paper]), SMAC ([Repo][smac_repo], [Paper][smac_paper]), [2D RoboCup][robocup], [Flatland][flatland], OpenSpiel ([Repo][openspiel_repo], [Paper][openspiel_paper]) environments, as well as a few custom [environments][debug] inside Mava.

With our integration with PettingZoo, we support popular Multi-Agent environments such as SISL ([Repo][sisl_repo], [Paper][sisl_paper]), MPE ([Repo][mpe_repo], [Paper][mpe_paper]) and [Multi-Agent Atari](https://www.pettingzoo.ml/atari) environments.

## Continuous control
We include a number of systems running on continuous control tasks.

### Debugging Environment - Simple Spread
-   **MADDPG**:
    a MADDPG system running on the continuous action space simple_spread MPE environment.
    - *Feedforward*:
        -  [decentralised](debugging/simple_spread/feedforward/decentralised/run_maddpg.py), [decentralised record agents](debugging/simple_spread/feedforward/decentralised/run_maddpg_record.py) (***Example recording agents acting in the environment***), [decentralised scaling](debugging/simple_spread/feedforward/decentralised/run_maddpg_scaling.py) (***Example scaling to 4 executors***), [decentralised custom loggers](debugging/simple_spread/feedforward/decentralised/run_maddpg_custom_logging.py) (***Example using custom logging***), [decentralised lr scheduling](debugging/simple_spread/feedforward/decentralised/run_maddpg_lr_schedule.py) (***Example using lr schedule***),
[centralised](debugging/simple_spread/feedforward/centralised/run_maddpg.py), [networked](debugging/simple_spread/feedforward/networked/run_maddpg.py) (***Example using a fully-connected, networked architecture***), [networked with custom architecture](debugging/simple_spread/feedforward/networked/run_maddpg_custom_network.py) (***Example using a custom, sparse, networked architecture***) and [state_based](debugging/simple_spread/feedforward/state_based/run_maddpg.py) .
    - *Recurrent*
        - [decentralised](debugging/simple_spread/recurrent/decentralised/run_maddpg.py) and [state_based](debugging/simple_spread/recurrent/state_based/run_maddpg.py).

-   **MAD4PG**:
    a MAD4PG system running on the continuous action space simple_spread MPE environment.
    - *Feedforward*
        - [decentralised](debugging/simple_spread/feedforward/decentralised/run_mad4pg.py), [centralised](debugging/simple_spread/feedforward/centralised/run_mad4pg.py)
    and [state_based](debugging/simple_spread/feedforward/state_based/run_mad4pg.py).
    - *Recurrent*
        - [decentralised](debugging/simple_spread/recurrent/decentralised/run_mad4pg.py).

### PettingZoo - Multiwalker
  -   **MADDPG**:
      a MADDPG system running on the Multiwalker environment.
      - *Feedforward*
        - [decentralised](petting_zoo/sisl/multiwalker/feedforward/decentralised/run_maddpg.py) and [centralised](petting_zoo/sisl/multiwalker/feedforward/centralised/run_maddpg.py).
      - *Recurrent*
        - [decentralised](petting_zoo/sisl/multiwalker/recurrent/decentralised/run_maddpg.py).

  -   **MAD4PG**:
      a MAD4PG system running on the Multiwalker environment.
      - *Feedforward*
        - [decentralised](petting_zoo/sisl/multiwalker/feedforward/decentralised/run_mad4pg.py) and [decentralised record agents](petting_zoo/sisl/multiwalker/feedforward/decentralised/run_mad4pg_record.py) (***Example recording agents acting in the environment***).

### 2D RoboCup
-   **MAD4PG**:
    a MAD4PG system running on the RoboCup environment.
    - *Recurrent*
      - [state_based](robocup/recurrent/state_based/run_mad4pg.py).
## Discrete control

We also include a number of systems running on discrete action space environments.

### Debugging Environment - Simple Spread
  -   **MAPPO**:
      a MAPPO system running on the discrete action space simple_spread MPE environment.
      - *Feedforward*
        - [decentralised](debugging/simple_spread/feedforward/decentralised/run_mappo.py) and [centralised](debugging/simple_spread/feedforward/centralised/run_mappo.py).

  -   **MADQN**:
      a MADQN system running on the discrete action space simple_spread MPE environment.
      - *Feedforward*
        - [decentralised](debugging/simple_spread/feedforward/decentralised/run_madqn.py) and [decentralised lr scheduling](debugging/simple_spread/feedforward/decentralised/run_madqn_lr_schedule.py) (***Example using lr schedule***).
      - *Recurrent*
        - [decentralised](debugging/simple_spread/recurrent/decentralised/run_madqn.py) and [decentralised with coms](debugging/simple_spread/recurrent/decentralised/run_madqn_with_coms.py) (***Example using a system with communication***).

  -   **QMIX**:
      a QMIX system running on the discrete action space simple_spread MPE environment.
      - *Feedforward*
        - [decentralised](debugging/simple_spread/feedforward/decentralised/run_qmix.py).

  -   **VDN**:
      a VDN system running on the discrete action space simple_spread MPE environment.
      - *Feedforward*
        - [decentralised](debugging/simple_spread/feedforward/decentralised/run_vdn.py).

  -   **DIAL**:
      a DIAL system running on the discrete action space simple_spread MPE environment.
      - *Recurrent*
        - [decentralised](debugging/simple_spread/recurrent/decentralised/run_dial.py).

### Debugging Environment - Switch
-    **DIAL**:
    a DIAL system running on the discrete custom SwitchGame environment.
     - *Recurrent*
        - [decentralised](debugging/switch/recurrent/decentralised/run_dial.py).

### PettingZoo - Multi-Agent Atari
-   **MADQN**:
   a MADQN system running on the two-player competitive Atari Pong environment.
    - *Feedforward*
      - [decentralised](petting_zoo/atari/pong/feedforward/decentralised/run_madqn.py).

### PettingZoo - Multi-Agent Particle Environment
  -   **MADDPG**:
      a MADDPG system running on the Simple Speaker Listener environment.
      - *Feedforward*
        - [ decentralised](petting_zoo/mpe/simple_speaker_listener/feedforward/decentralised/run_maddpg.py).

  -   **MADDPG**:
      a MADDPG system running on the Simple Spread environment.
      - *Feedforward*
        - [decentralised](petting_zoo/mpe/simple_spread/feedforward/decentralised/run_maddpg.py).

### SMAC - StarCraft Multi-Agent Challenge
-   **MADQN**:
    a MADQN system running on the SMAC environment.
    - *Feedforward*
      - [decentralised](smac/feedforward/decentralised/run_madqn.py).
    - *Recurrent*
      - [decentralised with custom agent networks](smac/recurrent/decentralised/run_madqn.py) (***Example using custom agent networks***).

-   **QMIX**:
    a QMIX system running on the SMAC environment.
    - *Feedforward*
      - [decentralised](smac/feedforward/decentralised/run_qmix.py).

-   **VDN**:
    a VDN system running on the SMAC environment.
    - *Feedforward*
      - [decentralised](smac/feedforward/decentralised/run_vdn.py) and [decentralised record agents](smac/feedforward/decentralised/run_vdn_record.py).

### OpenSpiel - Tic Tac Toe
  -   **MADQN**:
      a MADQN system running on the OpenSpiel environment.
      - *Feedforward*
        - [decentralised](openspiel/tic_tac_toe/feedforward/decentralised/run_madqn.py).


[debug]: ../mava/utils/debugging
[pz_repo]: https://github.com/PettingZoo-Team/PettingZoo
[pz_paper]: https://arxiv.org/abs/2009.14471
[flatland]: https://gitlab.aicrowd.com/flatland/flatland
[smac_repo]: https://github.com/oxwhirl/smac
[smac_paper]: https://arxiv.org/abs/1902.04043
[openspiel_repo]: https://github.com/deepmind/open_spiel
[openspiel_paper]: https://github.com/deepmind/open_spiel
[sisl_repo]: https://github.com/sisl/MADRL
[sisl_paper]: http://ala2017.it.nuigalway.ie/papers/ALA2017_Gupta.pdf
[mpe_repo]: https://github.com/openai/multiagent-particle-envs
[mpe_paper]: https://arxiv.org/abs/1706.02275
[robocup]: https://github.com/rcsoccersim
[quickstart]: ./quickstart.ipynb
