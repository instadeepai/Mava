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
IMAGE=instadeepct/mava:latest
DOCKER_RUN=docker run $(RUN_FLAGS) $(IMAGE)

# Set example to run when using `make run`
MADDPG=examples/debugging_envs/run_feedforward_maddpg.py
MADQN=examples/debugging_envs/run_feedforward_madqn.py
MAPPO=examples/debugging_envs/run_feedforward_mappo.py
QMIX=examples/debugging_envs/run_feedforward_qmix.py
VDN=examples/debugging_envs/run_feedforward_vdn.py

QMIX-PZ=examples/petting_zoo/run_feedforward_qmix.py
VDN-PZ==examples/petting_zoo/run_feedforward_vdn.py

# make file commands
run-maddpg:
	$(DOCKER_RUN) python $(MADDPG)

run-madqn:
	$(DOCKER_RUN) python $(MADQN)

run-mappo:
	$(DOCKER_RUN) python $(MAPPO)

run-qmix:
	$(DOCKER_RUN) python $(QMIX)

run-vdn:
	$(DOCKER_RUN) python $(VDN)

run-qmix-pz:
	$(DOCKER_RUN) python $(QMIX-PZ)

run-vdn-pz:
	$(DOCKER_RUN) python $(VDN-PZ)

bash:
	$(DOCKER_RUN) bash

record:
	$(DOCKER_RUN)  /bin/bash -c "./startup.sh ; python main.py $(MADDPG) "

build:
	docker build --tag $(IMAGE) .

push:
	docker login
	-docker push $(IMAGE)
	docker logout

pull:
	docker pull $(IMAGE)
