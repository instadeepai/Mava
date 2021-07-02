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
# Default example
EXAMPLE=examples/debugging/simple_spread/feedforward/decentralised/run_maddpg.py

# make file commands
run:
	$(DOCKER_RUN) python $(EXAMPLE) --base_dir /home/app/mava/logs/

bash:
	$(DOCKER_RUN) bash

run-tensorboard:
	$(DOCKER_RUN_TENSORBOARD) /bin/bash -c "  tensorboard --bind_all --logdir  /home/app/mava/logs/ & python $(EXAMPLE) --base_dir /home/app/mava/logs/; "

record:
	$(DOCKER_RUN)  /bin/bash -c "./startup.sh ; python $(MADDPG_RECORD) --base_dir /home/app/mava/logs/ "

build:
	docker build --tag $(IMAGE) .

build_robocup:
	docker build --tag $(IMAGE) -f ./Dockerfile.robocup .

build_sc2:
	docker build --tag $(IMAGE) -f ./Dockerfile.sc2 .

push:
	docker login
	-docker push $(IMAGE)
	docker logout

pull:
	docker pull $(IMAGE)
