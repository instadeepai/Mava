# Check if GPU is available
NVCC_RESULT := $(shell which nvcc 2> NULL)
NVCC_TEST := $(notdir $(NVCC_RESULT))
ifeq ($(NVCC_TEST),nvcc)
GPUS=--gpus all
else
GPUS=
endif

# For Windows use CURDIR
ifeq ($(PWD),)
PWD := $(CURDIR)
endif

# Set flag for docker run command
BASE_FLAGS=-it --rm  -v ${PWD}:/home/app/mava -w /home/app/mava
RUN_FLAGS=$(GPUS) $(BASE_FLAGS)
RUN_FLAGS_TENSORBOARD=$(GPUS) -p 6006:6006 $(BASE_FLAGS)

# Default version is jax-core
DOCKER_IMAGE_NAME = mava
DOCKER_IMAGE_TAG = mava-edu
IMAGE = $(DOCKER_IMAGE_NAME):$(DOCKER_IMAGE_TAG)-latest
DOCKER_RUN=docker run $(RUN_FLAGS) $(IMAGE)
DOCKER_RUN_TENSORBOARD=docker run $(RUN_FLAGS_TENSORBOARD) $(IMAGE)


# make file commands
build:
	DOCKER_BUILDKIT=1 docker build --tag $(IMAGE) --target $(DOCKER_IMAGE_TAG)  --build-arg record=$(record) .

run:
	$(DOCKER_RUN) python $(example) --base_dir /home/app/mava/logs/

run-tensorboard:
	$(DOCKER_RUN_TENSORBOARD) /bin/bash -c "  tensorboard --bind_all --logdir  /home/app/mava/logs/ & python $(example) --base_dir /home/app/mava/logs/; "

run-record:
	$(DOCKER_RUN)  /bin/bash -c "./bash_scripts/startup_screen.sh ; python $(example) --base_dir /home/app/mava/logs/ "

bash:
	$(DOCKER_RUN) bash

push:
	docker login
	-docker push $(IMAGE)
	docker logout

pull:
	docker pull $(IMAGE)
