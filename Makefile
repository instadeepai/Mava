# Check if GPU is available
NVCC_RESULT := $(shell which nvcc 2> NULL)
NVCC_TEST := $(notdir $(NVCC_RESULT))
ifeq ($(NVCC_TEST),nvcc)
GPUS=--gpus all
else
GPUS=
endif

# Set flag for docker run command
BASE_FLAGS=-it --rm  -v $(PWD):/home/app/mava -w /home/app/mava
RUN_FLAGS=$(GPUS) $(BASE_FLAGS)
IMAGE=mava:latest
RUN_FLAGS_TENSORBOARD=$(GPUS) -p 6006:6006 $(BASE_FLAGS)
DOCKER_RUN=docker run $(RUN_FLAGS) $(IMAGE)
DOCKER_RUN_TENSORBOARD=docker run $(RUN_FLAGS_TENSORBOARD) $(IMAGE)

# Set example to run when using `make run`
MADDPG=examples/debugging_envs/run_decentralised_feedforward_maddpg.py
MAPPO=examples/debugging_envs/run_decentralised_feedforward_mappo.py
MASAC=examples/debugging_envs/run_centralised_feedforward_masac.py
MADQN=examples/debugging_envs/run_feedforward_madqn.py
QMIX=examples/debugging_envs/run_feedforward_qmix.py
VDN=examples/debugging_envs/run_feedforward_vdn.py

MADDPG-PZ=examples/petting_zoo/run_decentralised_feedforward_maddpg.py
QMIX-PZ=examples/petting_zoo/run_feedforward_qmix.py
VDN-PZ==examples/petting_zoo/run_feedforward_vdn.py

MADDPG_RECORD=examples/petting_zoo/run_feedforward_maddpg_record_video.py

# make file commands
run:
	$(DOCKER_RUN) python $(EXAMPLE) --base_dir /home/app/mava/logs/

run-maddpg:
	$(DOCKER_RUN) python $(MADDPG) --base_dir /home/app/mava/logs/

run-masac:
	$(DOCKER_RUN) python $(MASAC)

run-madqn:
	$(DOCKER_RUN) python  $(MADQN) --base_dir /home/app/mava/logs/

run-mappo:
	$(DOCKER_RUN) python $(MAPPO) --base_dir /home/app/mava/logs/

run-qmix:
	$(DOCKER_RUN) python  $(QMIX) --base_dir /home/app/mava/logs/

run-vdn:
	$(DOCKER_RUN) python $(VDN) --base_dir /home/app/mava/logs/

run-maddpg-pz:
	$(DOCKER_RUN) python $(MADDPG-PZ) --base_dir /home/app/mava/logs/

run-qmix-pz:
	$(DOCKER_RUN) python $(QMIX-PZ) --base_dir /home/app/mava/logs/

run-vdn-pz:
	$(DOCKER_RUN) python $(VDN-PZ) --base_dir /home/app/mava/logs/

bash:
	$(DOCKER_RUN) bash

run-tensorboard:
	$(DOCKER_RUN_TENSORBOARD) /bin/bash -c "  tensorboard --bind_all --logdir  /home/app/mava/logs/ & python $(EXAMPLE) --base_dir /home/app/mava/logs/; "

record:
	$(DOCKER_RUN)  /bin/bash -c "./startup.sh ; python $(MADDPG_RECORD) --base_dir /home/app/mava/logs/ "

build:
	docker build --tag $(IMAGE) .

push:
	docker login
	-docker push $(IMAGE)
	docker logout

pull:
	docker pull $(IMAGE)
