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


[debug_ippo_ff_dec_jax]: https://github.com/instadeepai/Mava/blob/develop/examples/debugging/simple_spread/feedforward/decentralised/run_ippo.py
[debug_ippo_ff_dec_jax_record]: https://github.com/instadeepai/Mava/blob/develop/examples/debugging/simple_spread/feedforward/decentralised/run_ippo_with_monitoring.py
[debug_ippo_ff_dec_jax_single_process]: https://github.com/instadeepai/Mava/blob/develop/examples/debugging/simple_spread/feedforward/decentralised/run_ippo_single_process.py
[debug_ippo_ff_dec_jax_checkpoint]: https://github.com/instadeepai/Mava/blob/develop/examples/debugging/simple_spread/feedforward/decentralised/run_ippo_restore_checkpoint.py

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
[jax_quickstart]: https://github.com/instadeepai/Mava/blob/develop/examples/quickstart.ipynb

[debug_ippo_ff_dec_jax]: https://github.com/instadeepai/Mava/blob/develop/examples/debugging/simple_spread/feedforward/decentralised/run_ippo.py
[debug_ippo_ff_dec_jax_record]: https://github.com/instadeepai/Mava/blob/develop/examples/debugging/simple_spread/feedforward/decentralised/run_ippo_with_monitoring.py
[debug_ippo_ff_dec_jax_single_process]: https://github.com/instadeepai/Mava/blob/develop/examples/debugging/simple_spread/feedforward/decentralised/run_ippo_single_process.py
[debug_ippo_ff_dec_jax_checkpoint]: https://github.com/instadeepai/Mava/blob/develop/examples/debugging/simple_spread/feedforward/decentralised/run_ippo_restore_checkpoint.py
[flatland_ippo_ff_dec_jax]: https://github.com/instadeepai/Mava/blob/develop/examples/flatland/feedforward/decentralised/run_ippo.py
[pz_coop_pong_ippo_ff_dec_jax]: https://github.com/instadeepai/Mava/blob/develop/examples/petting_zoo/butterfly/cooperative_pong/feedforward/decentralised/run_ippo.py
[smac_ippo_ff_dec_jax]: https://github.com/instadeepai/Mava/blob/develop/examples/smac/feedforward/decentralised/run_ippo.py
