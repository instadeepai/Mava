# Before running this script, excute the "chmod u+x run_sweep.sh" then run the script using the tmux terminal.
# python mava/systems/ff_ippo_continuous.py system.seed=42,43,44 env.env_name=ant_4x2,halfcheetah_6x1,hopper_3x1,"humanoid_9|8",walker2d_2x3 -m && \


envs=(ant_4x2 halfcheetah_6x1 walker2d_2x3) # hopper_3x1,"humanoid_9|8
# seeds= (42 12)

for e in "${envs[@]}"
do
    # if [ "$e" == "ant_4x2" ] || [ "$e" == "walker2d_2x3" ]
    # then
    #     for seed in {42..44}
    #     do
    #         python mava/systems/ff_ippo_continuous.py system.seed="$seed" env.env_name="$e"  env.scenario.task_name="$e" env.kwargs.homogenisation_method=None -m &&
    #         python mava/systems/ff_mappo_continuous.py system.seed="$seed" env.env_name="$e"  env.scenario.task_name="$e" env.kwargs.homogenisation_method=None -m # &&
    #         # python mava/systems/rec_ippo_continuous.py system.seed="$seed" env.env_name="$e"  env.scenario.task_name="$e" env.kwargs.homogenisation_method=None -m &&
    #         # python mava/systems/rec_mappo_continuous.py system.seed="$seed" env.env_name="$e"  env.scenario.task_name="$e" env.kwargs.homogenisation_method=None -m
    #     done
    # else
    for seed in {42..44}
    do
        python mava/systems/ff_ippo_continuous.py system.seed="$seed" env.env_name="$e"  env.scenario.task_name="$e"  &&
        python mava/systems/ff_mappo_continuous.py system.seed="$seed" env.env_name="$e"  env.scenario.task_name="$e" &&
        python mava/systems/rec_ippo_continuous.py system.seed="$seed" env.env_name="$e"  env.scenario.task_name="$e"
        # python mava/systems/rec_mappo_continuous.py system.seed="$seed" env.env_name="$e"  env.scenario.task_name="$e"
    done
done
