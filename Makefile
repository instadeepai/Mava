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
version = jax-core
DOCKER_IMAGE_NAME = mava
DOCKER_IMAGE_TAG = $(version)
IMAGE = $(DOCKER_IMAGE_NAME):$(DOCKER_IMAGE_TAG)-latest
DOCKER_RUN=docker run $(RUN_FLAGS) $(IMAGE)
DOCKER_RUN_TENSORBOARD=docker run $(RUN_FLAGS_TENSORBOARD) $(IMAGE)

# Choose correct image for example
ifneq (,$(findstring jax,$(example)))
	ifneq (,$(findstring debugging,$(example)))
	DOCKER_IMAGE_TAG=jax-core
	else ifneq (,$(findstring petting,$(example)))
	DOCKER_IMAGE_TAG=pz-jax
	else ifneq (,$(findstring flatland,$(example)))
	DOCKER_IMAGE_TAG=flatland-jax
	else ifneq (,$(findstring openspiel,$(example)))
	DOCKER_IMAGE_TAG=openspiel-jax
	else ifneq (,$(findstring robocup,$(example)))
	DOCKER_IMAGE_TAG=robocup-jax
	else ifneq (,$(findstring smac,$(example)))
	DOCKER_IMAGE_TAG=sc2-jax
	else ifneq (,$(findstring meltingpot,$(example)))
	DOCKER_IMAGE_TAG=meltingpot-jax
	endif
else
	ifneq (,$(findstring debugging,$(example)))
	DOCKER_IMAGE_TAG=tf-core
	else ifneq (,$(findstring petting,$(example)))
	DOCKER_IMAGE_TAG=pz
	else ifneq (,$(findstring flatland,$(example)))
	DOCKER_IMAGE_TAG=flatland
	else ifneq (,$(findstring openspiel,$(example)))
	DOCKER_IMAGE_TAG=openspiel
	else ifneq (,$(findstring robocup,$(example)))
	DOCKER_IMAGE_TAG=robocup
	else ifneq (,$(findstring smac,$(example)))
	DOCKER_IMAGE_TAG=sc2
	else ifneq (,$(findstring meltingpot,$(example)))
	DOCKER_IMAGE_TAG=meltingpot
	endif
endif

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

run-tests:
	$(DOCKER_RUN) /bin/bash bash_scripts/local_tests.sh

run-integration-tests:
	$(DOCKER_RUN) /bin/bash bash_scripts/local_tests.sh true

run-checks:
	$(DOCKER_RUN) /bin/bash bash_scripts/check_format.sh

push:
	docker login
	-docker push $(IMAGE)
	docker logout

pull:
	docker pull $(IMAGE)
