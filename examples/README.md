# Examples

We include a non-exhaustive number of examples for Jax-based systems, showing common use-cases for Mava. We also have a [quickstart notebook][quickstart] that can be used to quickly create and train your first multi-agent system.

# Examples
We also have various Jax-based examples which make use of the callback design paradigm. Our pre-implemented Jax-based systems are continually expanding so please check back often to see new implemented systems.

## Discrete control

### Debugging Environment - Simple Spread

- **IPPO**:
    an IPPO system running on the discrete action space simple_spread MPE environment.
  - *Feedforward*
    - [decentralised][debug_ippo_ff_dec_jax]
    - [decentralised record agents][debug_ippo_ff_dec_jax_record] (***recording agents acting in the environment***).
    - [decentralised single process][debug_ippo_ff_dec_jax_single_process] (***running in single process mode***).
    - [decentralised restore checkpoint][debug_ippo_ff_dec_jax_checkpoint] (***continuing training by restoring from an existing checkpoint***).
    - [decentralised evaluation intervals][debug_ippo_ff_dec_eval_intevals_jax] (***perform evaluation at custom intervals for custom durations***)


### Flatland

- **IPPO**:
    an IPPO system running on the discrete action space flatland environment.
  - *Feedforward*
    - [decentralised][flatland_ippo_ff_dec_jax]

### Pettingzoo - Cooperative pong

- **IPPO**:
    an IPPO system running on the discrete action space Cooperative pong MPE environment.
  - *Feedforward*
    - [decentralised][pz_coop_pong_ippo_ff_dec_jax]

### Pettingzoo - Multi-Agent Particle Environment

- **IPPO**:
    an IPPO system running on the Simple Spread environment.
  - *Feedforward*
    - [decentralised][pz_simple_spread]

### SMAC - StarCraft Multi-Agent Challenge

- **IPPO**:
    an IPPO system running on the discrete action space 3m SMAC environment.
  - *Feedforward*
    - [decentralised][smac_ippo_ff_dec_jax]
    - [decentralised evaluation intervals][smac_ippo_ff_dec_eval_intervals_jax] (***perform evaluation at custom intervals for custom durations***)



<!-- Examples -->
[quickstart]: https://github.com/instadeepai/Mava/blob/develop/examples/quickstart.ipynb

[debug_ippo_ff_dec_jax]: https://github.com/instadeepai/Mava/blob/develop/examples/debugging/simple_spread/feedforward/decentralised/run_ippo.py
[debug_ippo_ff_dec_jax_record]: https://github.com/instadeepai/Mava/blob/develop/examples/debugging/simple_spread/feedforward/decentralised/run_ippo_with_monitoring.py
[debug_ippo_ff_dec_jax_single_process]: https://github.com/instadeepai/Mava/blob/develop/examples/debugging/simple_spread/feedforward/decentralised/run_ippo_single_process.py
[debug_ippo_ff_dec_jax_checkpoint]: https://github.com/instadeepai/Mava/blob/develop/examples/debugging/simple_spread/feedforward/decentralised/run_ippo_restore_checkpoint.py
[flatland_ippo_ff_dec_jax]: https://github.com/instadeepai/Mava/blob/develop/examples/flatland/feedforward/decentralised/run_ippo.py
[pz_coop_pong_ippo_ff_dec_jax]: https://github.com/instadeepai/Mava/blob/develop/examples/petting_zoo/butterfly/cooperative_pong/feedforward/decentralised/run_ippo.py
[smac_ippo_ff_dec_jax]: https://github.com/instadeepai/Mava/blob/develop/examples/smac/feedforward/decentralised/run_ippo.py
[debug_ippo_ff_dec_eval_intevals_jax]: https://github.com/instadeepai/Mava/blob/develop/examples/debugging/simple_spread/feedforward/decentralised/run_ippo_eval_intervals.py
[smac_ippo_ff_dec_eval_intervals_jax]: https://github.com/instadeepai/Mava/blob/develop/examples/smac/feedforward/decentralised/run_ippo_eval_intervals.py
[pz_simple_spread]: https://github.com/instadeepai/Mava/blob/develop/examples/petting_zoo/simple_spread/feedforward/decentralised/run_ippo.py
