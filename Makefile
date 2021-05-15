# Check if GPU is available
NVCC_RESULT := $(shell which nvcc 2> NULL)
NVCC_TEST := $(notdir $(NVCC_RESULT))
ifeq ($(NVCC_TEST),nvcc)
GPUS=--gpus all
else
GPUS=
endif

# Setup fake screen
FAKE_DISPLAY := $(shell xvfb-run -s "-screen 0 1400x900x24" bash)

# Set flag for docker run command
BASE_FLAGS=-it --rm  -v $(PWD):/home/app/mava -w /home/app/mava
RUN_FLAGS=$(GPUS) $(BASE_FLAGS)
IMAGE=instadeepct/mava:latest
DOCKER_RUN=docker run $(RUN_FLAGS) $(IMAGE)

# Set example to run when using `make run`
MADDPG=examples/debugging_envs/run_feedforward_maddpg.py
MADQN=examples/debugging_envs/run_feedforward_madqn.py
QMIX=examples/debugging_envs/run_feedforward_qmix.py

# make file commands
run-maddpg:
	$(DOCKER_RUN) python $(MADDPG)

run-madqn:
	$(DOCKER_RUN) python $(MADQN)

run-qmix:
	$(DOCKER_RUN) python $(QMIX)

run-vdn:
	$(DOCKER_RUN) python $(VDN)

run-mappo:
	$(DOCKER_RUN) python $(MAPPO)

bash:
	$(DOCKER_RUN) bash

build:
	docker build --tag $(IMAGE) .

push:
	docker login
	-docker push $(IMAGE)
	docker logout

pull:
	docker pull $(IMAGE)
