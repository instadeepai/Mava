#!/bin/bash
python mava/systems/rec_ippo.py -m system.seed=0,1,2,3,4,5,6,7,8,9 env.scenario=2s3z,3s5z,5m_vs_6m,10m_vs_11m,3s5z_vs_3s6z,3s_vs_5z,6h_vs_8z && python mava/systems/rec_mappo.py -m system.seed=0,1,2,3,4,5,6,7,8,9 env.scenario=2s3z,3s5z,5m_vs_6m,10m_vs_11m,3s5z_vs_3s6z,3s_vs_5z,6h_vs_8z
#python mava/systems/rec_ippo.py -m env.scenario=3s5z_vs_3s6z,3s_vs_5z,6h_vs_8z && python mava/systems/rec_mappo.py -m env.scenario=2s3z,3s5z,5m_vs_6m,10m_vs_11m,3s5z_vs_3s6z,3s_vs_5z,6h_vs_8z
