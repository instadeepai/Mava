# Examples

We include a non-exhaustive number of examples, showing common use-cases for Mava. We also have a [Quickstart notebook][quickstart] that can be used to quickly create and train your first Multi-Agent System.

## Continuous control

We include a number of systems running on continuous control tasks.

### Debugging Environment - Simple Spread

- **MADDPG**:
    a MADDPG system running on the continuous action space simple_spread MPE environment.
  - *Feedforward*:
    - decentralised
      - [decentralised][debug_maddpg_ff_dec]
      - [decentralised record agents][debug_maddpg_ff_dec_record]  (***recording agents acting in the environment***)
      - [decentralised executor scaling][debug_maddpg_ff_dec_scaling_executors] (***scaling to 4 executors***)
      - [decentralised multiple trainers][debug_maddpg_ff_dec_scaling_trainers](***using multiple trainers***)
      - [decentralised custom loggers][debug_maddpg_ff_dec_custom_logging] (***using custom logging***)
      - [decentralised lr scheduling][debug_maddpg_ff_dec_lr_scheduling](***using lr schedule***)
      - [decentralised evaluator intervals][debug_maddpg_ff_dec_eval_intervals](***running the evaluation loop at intervals***)

  - [centralised][debug_maddpg_cen] , [networked][debug_maddpg_networked] (***using a fully-connected, networked architecture***), [networked with custom architecture][debug_maddpg_networked_custom] (***using a custom, sparse, networked architecture***) and [state_based][debug_maddpg_state_based].

  - *Recurrent*
    - [decentralised][debug_maddpg_rec_dec] and [state_based][debug_maddpg_state_based].

- **MAD4PG**:
    a MAD4PG system running on the continuous action space simple_spread MPE environment.
  - *Feedforward*
    - [decentralised][debug_mad4pg_ff_dec], [centralised][debug_mad4pg_ff_cen]
    and [state_based][debug_mad4pg_ff_state_based].
  - *Recurrent*
    - [decentralised][debug_mad4pg_rec_dec].

### PettingZoo - Multiwalker

- **MADDPG**:
      a MADDPG system running on the Multiwalker environment.
  - *Feedforward*
    - [decentralised][pz_maddpg_ff_dec] and [centralised][pz_maddpg_ff_cen].
  - *Recurrent*
    - [decentralised][pz_maddpg_rec_dec].

- **MAD4PG**:
      a MAD4PG system running on the Multiwalker environment.
  - *Feedforward*
    - [decentralised][pz_mad4pg_ff_dec] and [decentralised record agents][pz_mad4pg_ff_dec_record] (***recording agents acting in the environment***).

### 2D RoboCup

- **MAD4PG**:
    a MAD4PG system running on the RoboCup environment.
  - *Recurrent* [state_based][robocup_mad4pg_ff_state_based].

## Discrete control

We also include a number of systems running on discrete action space environments.

### Debugging Environment - Simple Spread

- **MAPPO**:
      a MAPPO system running on the discrete action space simple_spread MPE environment.
  - *Feedforward*
    - [decentralised][debug_mappo_ff_dec] and [centralised][debug_mappo_ff_cen].

- **MADQN**:
      a MADQN system running on the discrete action space simple_spread MPE environment.
  - *Feedforward*
    - [decentralised][debug_madqn_ff_dec], [decentralised lr scheduling][debug_madqn_ff_dec_lr_schedule] (***using lr schedule***), [decentralised custom lr scheduling][debug_madqn_ff_dec_custom_lr_schedule] (***using custom lr schedule***) and [decentralised custom epsilon decay scheduling][debug_madqn_ff_dec_custom_eps_schedule] (***using configurable epsilon scheduling***).
  - *Recurrent*
    - [decentralised][debug_madqn_rec_dec].

- **VDN**:
      a VDN system running on the discrete action space simple_spread MPE environment.
  - *Recurrent* [centralised][debug_vdn_rec_cen].

### PettingZoo - Multi-Agent Atari

- **MADQN**:
   a MADQN system running on the two-player competitive Atari Pong environment.
  - *Recurrent* [decentralised][pz_madqn_pong_ff_dec].

### PettingZoo - Multi-Agent Particle Environment

- **MADDPG**:
      a MADDPG system running on the Simple Speaker Listener environment.
  - *Feedforward* [decentralised][pz_maddpg_mpe_ssl_ff_dec].

- **MADDPG**:
      a MADDPG system running on the Simple Spread environment.
  - *Feedforward* [decentralised][pz_maddpg_mpe_ss_ff_dec].

### SMAC - StarCraft Multi-Agent Challenge

- **MADQN**:
    a MADQN system running on the SMAC environment.
  - *Feedforward*
    - [decentralised][smac_madqn_ff_dec].
  - *Recurrent*
    - [decentralised][smac_madqn_rec_dec].

- **QMIX**:
    a QMIX system running on the SMAC environment.
  - *Recurrent* [centralised][smac_qmix_rec_cen].

- **VDN**:
    a VDN system running on the SMAC environment.
  - *Recurrent* [centralised][smac_vdn_rec_cen].

### OpenSpiel - Tic Tac Toe

- **MADQN**:
      a MADQN system running on the OpenSpiel environment.
  - *Feedforward* [decentralised][openspiel_madqn_ff_dec].

<!-- Examples -->
[quickstart]: https://github.com/instadeepai/Mava/blob/develop/examples/tf/quickstart.ipynb
<!-- Continous -->
[debug_maddpg_ff_dec]: https://github.com/instadeepai/Mava/blob/develop/examples/tf/debugging/simple_spread/feedforward/decentralised/run_maddpg.py
[debug_maddpg_ff_dec_record]: https://github.com/instadeepai/Mava/blob/develop/examples/tf/debugging/simple_spread/feedforward/decentralised/run_maddpg_record.py
[debug_maddpg_ff_dec_scaling_executors]: https://github.com/instadeepai/Mava/blob/develop/examples/tf/debugging/simple_spread/feedforward/decentralised/run_maddpg_scale_executors.py
[debug_maddpg_ff_dec_scaling_trainers]: https://github.com/instadeepai/Mava/blob/develop/examples/tf/debugging/simple_spread/feedforward/decentralised/run_maddpg_scale_trainers.py
[debug_maddpg_ff_dec_custom_logging]: https://github.com/instadeepai/Mava/blob/develop/examples/tf/debugging/simple_spread/feedforward/decentralised/run_maddpg_custom_logging.py
[debug_maddpg_ff_dec_lr_scheduling]: https://github.com/instadeepai/Mava/blob/develop/examples/tf/debugging/simple_spread/feedforward/decentralised/run_maddpg_lr_schedule.py
[debug_maddpg_ff_dec_eval_intervals]: https://github.com/instadeepai/Mava/blob/develop/examples/tf/debugging/simple_spread/feedforward/decentralised/run_mad4pg_evaluator_interval.py
[debug_maddpg_cen]: https://github.com/instadeepai/Mava/blob/develop/examples/tf/debugging/simple_spread/feedforward/centralised/run_maddpg.py
[debug_maddpg_networked]: https://github.com/instadeepai/Mava/blob/develop/examples/tf/debugging/simple_spread/feedforward/networked/run_maddpg.py
[debug_maddpg_networked_custom]: https://github.com/instadeepai/Mava/blob/develop/examples/tf/debugging/simple_spread/feedforward/networked/run_maddpg_custom_network.py
[debug_maddpg_state_based]: https://github.com/instadeepai/Mava/blob/develop/examples/tf/debugging/simple_spread/feedforward/state_based/run_maddpg.py
[debug_maddpg_rec_dec]: https://github.com/instadeepai/Mava/blob/develop/examples/tf/debugging/simple_spread/recurrent/decentralised/run_maddpg.py
[debug_maddpg_state_based]: https://github.com/instadeepai/Mava/blob/develop/examples/tf/debugging/simple_spread/recurrent/state_based/run_maddpg.py

[debug_mad4pg_ff_dec]: https://github.com/instadeepai/Mava/blob/develop/examples/tf/debugging/simple_spread/feedforward/decentralised/run_mad4pg.py
[debug_mad4pg_ff_cen]:  https://github.com/instadeepai/Mava/blob/develop/examples/tf/debugging/simple_spread/feedforward/centralised/run_mad4pg.py
[debug_mad4pg_ff_state_based]: https://github.com/instadeepai/Mava/blob/develop/examples/tf/debugging/simple_spread/feedforward/state_based/run_mad4pg.py
[debug_mad4pg_rec_dec]: https://github.com/instadeepai/Mava/blob/develop/examples/tf/debugging/simple_spread/recurrent/decentralised/run_mad4pg.py

[pz_maddpg_ff_dec]: https://github.com/instadeepai/Mava/blob/develop/examples/tf/petting_zoo/sisl/multiwalker/feedforward/decentralised/run_maddpg.py
[pz_maddpg_ff_cen]: https://github.com/instadeepai/Mava/blob/develop/examples/tf/petting_zoo/sisl/multiwalker/feedforward/centralised/run_maddpg.py
[pz_maddpg_rec_dec]: https://github.com/instadeepai/Mava/blob/develop/examples/tf/petting_zoo/sisl/multiwalker/recurrent/decentralised/run_maddpg.py

[pz_mad4pg_ff_dec]: https://github.com/instadeepai/Mava/blob/develop/examples/tf/petting_zoo/sisl/multiwalker/feedforward/decentralised/run_mad4pg.py
[pz_mad4pg_ff_dec_record]: https://github.com/instadeepai/Mava/blob/develop/examples/tf/petting_zoo/sisl/multiwalker/feedforward/decentralised/run_mad4pg_record.py

[robocup_mad4pg_ff_state_based]: https://github.com/instadeepai/Mava/blob/develop/examples/tf/robocup/recurrent/state_based/run_mad4pg.py

<!-- Discrete -->
[debug_mappo_ff_dec]: https://github.com/instadeepai/Mava/blob/develop/examples/tf/debugging/simple_spread/feedforward/decentralised/run_mappo.py
[debug_mappo_ff_cen]: https://github.com/instadeepai/Mava/blob/develop/examples/tf/debugging/simple_spread/feedforward/centralised/run_mappo.py

[debug_madqn_ff_dec]: https://github.com/instadeepai/Mava/blob/develop/examples/tf/debugging/simple_spread/feedforward/decentralised/run_madqn.py
[debug_madqn_ff_dec_lr_schedule]: https://github.com/instadeepai/Mava/blob/develop/examples/tf/debugging/simple_spread/feedforward/decentralised/run_madqn_lr_schedule.py
[debug_madqn_ff_dec_custom_lr_schedule]: https://github.com/instadeepai/Mava/blob/develop/examples/tf/debugging/simple_spread/feedforward/decentralised/run_madqn_custom_lr_schedule.py
[debug_madqn_ff_dec_custom_eps_schedule]: https://github.com/instadeepai/Mava/blob/develop/examples/tf/debugging/simple_spread/feedforward/decentralised/run_madqn_configurable_epsilon.py
[debug_madqn_rec_dec]: https://github.com/instadeepai/Mava/blob/develop/examples/tf/debugging/simple_spread/recurrent/decentralised/run_madqn.py

[debug_vdn_rec_cen]: https://github.com/instadeepai/Mava/blob/develop/examples/tf/debugging/simple_spread/recurrent/centralised/run_vdn.py

[pz_madqn_pong_rec_dec]: https://github.com/instadeepai/Mava/blob/develop/examples/tf/petting_zoo/atari/pong/recurrent/centralised/run_madqn.py

[pz_maddpg_mpe_ssl_ff_dec]: https://github.com/instadeepai/Mava/blob/develop/examples/tf/petting_zoo/mpe/simple_speaker_listener/feedforward/decentralised/run_maddpg.py

[pz_maddpg_mpe_ss_ff_dec]: https://github.com/instadeepai/Mava/blob/develop/examples/tf/petting_zoo/mpe/simple_spread/feedforward/decentralised/run_maddpg.py

[smac_madqn_rec_dec]: https://github.com/instadeepai/Mava/blob/develop/examples/tf/smac/recurrent/decentralised/run_madqn.py

[smac_qmix_rec_cen]: https://github.com/instadeepai/Mava/blob/develop/examples/tf/smac/recurrent/centralised/run_qmix.py

[smac_vdn_rec_cen]: https://github.com/instadeepai/Mava/blob/develop/examples/tf/smac/recurrent/centralised/run_vdn.py

[openspiel_madqn_ff_dec]: https://github.com/instadeepai/Mava/blob/develop/examples/tf/openspiel/tic_tac_toe/feedforward/decentralised/run_madqn.py
