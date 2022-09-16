# Examples

We include a non-exhaustive number of examples for both Jax-based systems and Tensorflow-based systems, showing common use-cases for Mava. We also have a [Tensorflow quickstart notebook][tf_quickstart] and [Jax quickstart notebook][jax_quickstart] that can be used to quickly create and train your first multi-agent system.

# Jax examples
We also have various Jax-based examples which make use of the callback design paradigm. Our pre-implemented Jax-based systems are continually expanding so please check back often to see new implemented systems.

## Discrete control

### Debugging Environment - Simple Spread

- **IPPO**:
    an IPPO system running on the discrete action space simple_spread MPE environment.
  - *Feedforward*
    - [decentralised][debug_ippo_ff_dec_jax]
    - [decentralised record agents][debug_ippo_ff_dec_jax_record] (***recording agents acting in the environment***).
    - [decentralised single process][debug_ippo_ff_dec_jax_single_process] (***running in single process mode***).
    - [decentralised resotore checkpoint][debug_ippo_ff_dec_jax_checkpoint] (***continuing training by restoring from an existing checkpoint***).


[debug_ippo_ff_dec_jax]: https://github.com/instadeepai/Mava/blob/develop/examples/jax/debugging/simple_spread/feedforward/decentralised/run_ippo.py
[debug_ippo_ff_dec_jax_record]: https://github.com/instadeepai/Mava/blob/develop/examples/jax/debugging/simple_spread/feedforward/decentralised/run_ippo_with_monitoring.py
[debug_ippo_ff_dec_jax_single_process]: https://github.com/instadeepai/Mava/blob/develop/examples/jax/debugging/simple_spread/feedforward/decentralised/run_ippo_single_process.py
[debug_ippo_ff_dec_jax_checkpoint]: https://github.com/instadeepai/Mava/blob/develop/examples/jax/debugging/simple_spread/feedforward/decentralised/run_ippo_restore_checkpoint.py

### Flatland

- **IPPO**:
    an IPPO system running on the discrete action space simple_spread MPE environment.
  - *Feedforward*
    - [decentralised][flatland_ippo_ff_dec_jax]

### Pettingzoo - Cooperative pong

- **IPPO**:
    an IPPO system running on the discrete action space simple_spread MPE environment.
  - *Feedforward*
    - [decentralised][pz_coop_pong_ippo_ff_dec_jax]

### SMAC - StarCraft Multi-Agent Challenge

- **IPPO**:
    an IPPO system running on the discrete action space simple_spread MPE environment.
  - *Feedforward*
    - [decentralised][smac_ippo_ff_dec_jax]



<!-- Examples -->
[tf_quickstart]: https://github.com/instadeepai/Mava/blob/develop/examples/tf/quickstart.ipynb
[jax_quickstart]: https://github.com/instadeepai/Mava/blob/develop/examples/jax/quickstart.ipynb
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

[pz_mad4pg_ff_dec]: https://github.com/instadeepai/Mava/blob/develop/examples/tf/petting_zoo/sisl/multiwalker/feedforward/decentralised/run_mad4pg.py
[pz_mad4pg_ff_dec_record]: https://github.com/instadeepai/Mava/blob/develop/examples/tf/petting_zoo/sisl/multiwalker/feedforward/decentralised/run_mad4pg_record.py
[pz_mappo_ff_dec]: https://github.com/instadeepai/Mava/blob/develop/examples/tf/petting_zoo/sisl/multiwalker/feedforward/decentralised/run_mappo.py

[robocup_mad4pg_ff_state_based]:https://github.com/instadeepai/Mava/blob/develop/examples/tf/robocup/recurrent/state_based/run_mad4pg.py
<!-- Discrete -->
[debug_mappo_ff_dec]: https://github.com/instadeepai/Mava/blob/develop/examples/tf/debugging/simple_spread/feedforward/decentralised/run_mappo.py
[debug_mappo_ff_cen]: https://github.com/instadeepai/Mava/blob/develop/examples/tf/debugging/simple_spread/feedforward/centralised/run_mappo.py

[debug_madqn_ff_dec]: https://github.com/instadeepai/Mava/blob/develop/examples/tf/debugging/simple_spread/feedforward/decentralised/run_madqn.py
[debug_madqn_ff_dec_lr_schedule]: https://github.com/instadeepai/Mava/blob/develop/examples/tf/debugging/simple_spread/feedforward/decentralised/run_madqn_lr_schedule.py
[debug_madqn_ff_dec_custom_lr_schedule]: https://github.com/instadeepai/Mava/blob/develop/examples/tf/debugging/simple_spread/feedforward/decentralised/run_madqn_custom_lr_schedule.py
[debug_madqn_ff_dec_custom_eps_schedule]: https://github.com/instadeepai/Mava/blob/develop/examples/tf/debugging/simple_spread/feedforward/decentralised/run_madqn_configurable_epsilon.py
[debug_madqn_rec_dec]: https://github.com/instadeepai/Mava/blob/develop/examples/tf/debugging/simple_spread/recurrent/decentralised/run_madqn.py

[debug_vdn_rec_cen]: https://github.com/instadeepai/Mava/blob/develop/examples/tf/debugging/simple_spread/recurrent/centralised/run_vdn.py

[pz_madqn_pong_rec_dec]: https://github.com/instadeepai/Mava/blob/develop/examples/tf/petting_zoo/atari/pong/recurrent/decentralised/run_madqn.py

[pz_mappo_coop_pong_ff_dec]: https://github.com/instadeepai/Mava/blob/develop/examples/tf/petting_zoo/butterfly/cooperative_pong/feedforward/decentralised/run_mappo.py

[pz_maddpg_mpe_ssl_ff_dec]: https://github.com/instadeepai/Mava/blob/develop/examples/tf/petting_zoo/mpe/simple_speaker_listener/feedforward/decentralised/run_maddpg.py

[pz_maddpg_mpe_ss_ff_dec]: https://github.com/instadeepai/Mava/blob/develop/examples/tf/petting_zoo/mpe/simple_spread/feedforward/decentralised/run_maddpg.py

[smac_madqn_ff_dec]: https://github.com/instadeepai/Mava/blob/develop/examples/tf/smac/feedforward/decentralised/run_madqn.py
[smac_madqn_rec_dec]: https://github.com/instadeepai/Mava/blob/develop/examples/tf/smac/recurrent/decentralised/run_madqn.py

[smac_qmix_rec_cen]: https://github.com/instadeepai/Mava/blob/develop/examples/tf/smac/recurrent/centralised/run_qmix.py

[smac_vdn_rec_cen]: https://github.com/instadeepai/Mava/blob/develop/examples/tf/smac/recurrent/centralised/run_vdn.py

[openspiel_madqn_ff_dec]: https://github.com/instadeepai/Mava/blob/develop/examples/tf/openspiel/tic_tac_toe/feedforward/decentralised/run_madqn.py

[debug_ippo_ff_dec_jax]: https://github.com/instadeepai/Mava/blob/develop/examples/jax/debugging/simple_spread/feedforward/decentralised/run_ippo.py
[debug_ippo_ff_dec_jax_record]: https://github.com/instadeepai/Mava/blob/develop/examples/jax/debugging/simple_spread/feedforward/decentralised/run_ippo_with_monitoring.py
[debug_ippo_ff_dec_jax_single_process]: https://github.com/instadeepai/Mava/blob/develop/examples/jax/debugging/simple_spread/feedforward/decentralised/run_ippo_single_process.py
[debug_ippo_ff_dec_jax_checkpoint]: https://github.com/instadeepai/Mava/blob/develop/examples/jax/debugging/simple_spread/feedforward/decentralised/run_ippo_restore_checkpoint.py
[flatland_ippo_ff_dec_jax]: https://github.com/instadeepai/Mava/blob/develop/examples/jax/flatland/feedforward/decentralised/run_ippo.py
[pz_coop_pong_ippo_ff_dec_jax]: https://github.com/instadeepai/Mava/blob/develop/examples/jax/petting_zoo/butterfly/cooperative_pong/feedforward/decentralised/run_ippo.py
[smac_ippo_ff_dec_jax]: https://github.com/instadeepai/Mava/blob/develop/examples/jax/smac/feedforward/decentralised/run_ippo.py
