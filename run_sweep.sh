# Attention: Before running this script, excute the "chmod u+x run_sweep.sh" then run the script using the tmux terminal session.

# Note: `RUN1`: Run a sweep over systems and different tasks

# python mava/systems/ff_ippo_continuous.py system.seed=42,43,44 env.env_name=ant_4x2,walker2d_2x3 env.kwargs.homogenisation_method=None -m && \
# python mava/systems/ff_mappo_continuous.py system.seed=42,43,44 env.env_name=ant_4x2,walker2d_2x3 env.kwargs.homogenisation_method=None  -m

envs=(ant_4x2 halfcheetah_6x1 walker2d_2x3 hopper_3x1 "humanoid_9|8")

for e in "${envs[@]}"
do
    if [ "$e" == "ant_4x2" ] || [ "$e" == "walker2d_2x3" ]
    then
        for seed in {42..44}
        do
            python mava/systems/ff_ippo_continuous.py system.seed="$seed" env.env_name="$e"  env.scenario.task_name="$e"  env.kwargs.homogenisation_method="max" -m env.kwargs.homogenisation_method=None &&
            python mava/systems/ff_mappo_continuous.py system.seed="$seed" env.env_name="$e"  env.scenario.task_name="$e"  env.kwargs.homogenisation_method="max" -m env.kwargs.homogenisation_method=None &&
            python mava/systems/rec_ippo_continuous.py system.seed="$seed" env.env_name="$e"  env.scenario.task_name="$e"  env.kwargs.homogenisation_method="max" -m env.kwargs.homogenisation_method=None &&
            python mava/systems/rec_mappo_continuous.py system.seed="$seed" env.env_name="$e"  env.scenario.task_name="$e"  env.kwargs.homogenisation_method="max" -m env.kwargs.homogenisation_method=None
        done
    else
        for seed in {42..44}
        do
            python mava/systems/ff_ippo_continuous.py system.seed="$seed" env.env_name="$e"  env.scenario.task_name="$e"  env.kwargs.homogenisation_method="max" -m&&
            python mava/systems/ff_mappo_continuous.py system.seed="$seed" env.env_name="$e"  env.scenario.task_name="$e"  env.kwargs.homogenisation_method="max" -m&&
            python mava/systems/rec_ippo_continuous.py system.seed="$seed" env.env_name="$e"  env.scenario.task_name="$e"  env.kwargs.homogenisation_method="max" -m&&
            python mava/systems/rec_mappo_continuous.py system.seed="$seed" env.env_name="$e"  env.scenario.task_name="$e"  env.kwargs.homogenisation_method="max" -m
    fi
done

# Note `RUN2`: To run a hyperparamter sweep over one default task (run line1) or all tasks run (run line2):
#line1: python mava/systems/rec_ippo_continuous.py -m  # or --multirun
#line2: python mava/systems/rec_mappo_continuous.py env.env_name=ant_4x2,halfcheetah_6x1,hopper_3x1,"humanoid_9|8",walker2d_2x3 -m
