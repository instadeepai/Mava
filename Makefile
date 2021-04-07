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
EXAMPLE=examples/petting_zoo/run_maddpg.py


# make file commands
run:
	$(DOCKER_RUN) python $(EXAMPLE)

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
