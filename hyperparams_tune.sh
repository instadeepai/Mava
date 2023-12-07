#!/bin/bash
python mava/systems/ff_ippo.py

python mava/systems/ff_ippo.py -m arch.num_envs=64

python mava/systems/ff_ippo.py -m arch.num_envs=64 system.actor_lr=0.004 system.critic_lr=0.004 system.ppo_epochs=2 system.ent_coef=0.0

python mava/systems/ff_mappo.py

python mava/systems/ff_mappo.py -m arch.num_envs=64

python mava/systems/ff_mappo.py -m arch.num_envs=64 system.actor_lr=0.004 system.critic_lr=0.004 system.ppo_epochs=2 system.ent_coef=0.0
