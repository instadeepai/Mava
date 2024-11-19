# python mava/systems/ppo/anakin/ff_ippo.py -m env=rware system.num_updates=610 && \
# python mava/systems/ppo/anakin/ff_mappo.py -m env=rware system.num_updates=610 && \
# python mava/systems/ppo/anakin/rec_ippo.py -m env=rware system.num_updates=610 && \
python mava/systems/ppo/anakin/rec_mappo.py -m env=rware system.num_updates=610 && \
# python mava/systems/sable/anakin/ff_sable.py -m env=rware system.num_updates=610 && \
# python mava/systems/sable/anakin/rec_sable.py -m env=rware system.num_updates=610 && \
# python mava/systems/ppo/anakin/ff_cent_ppo.py -m env=rware system.num_updates=610 && \
python mava/systems/mat/anakin/mat.py -m env=rware system.num_updates=610 
