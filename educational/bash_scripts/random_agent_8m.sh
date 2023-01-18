# Script to run random agent on 8m
# Optional param to change base_dir - mainly for tensorboard. 
# Usage ./bash_scripts/random_agent_8m.sh ~/custom_folder
base_dir=${1:-'~/mava'}
python systems/random_agent.py --env_type smac --env_name 8m --max_total_steps=2000000 --base_dir $base_dir
