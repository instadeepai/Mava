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
QMIX=examples/debugging_envs/run_feedforward_qmix.py
QMIX-PZ=examples/petting_zoo/run_feedforward_qmix.py
MASAC=examples/debugging_envs/run_feedforward_masac.py

# make file commands
run-maddpg:
	$(DOCKER_RUN) python $(MADDPG)

run-masac:
	$(DOCKER_RUN) python $(MASAC)

run-madqn:
	$(DOCKER_RUN) python $(MADQN)

run-qmix:
	$(DOCKER_RUN) python $(QMIX)

run-qmix-pz:
	$(DOCKER_RUN) python $(QMIX-PZ)

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
