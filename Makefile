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

DOCKER_IMAGE_NAME = mava
IMAGE = $(DOCKER_IMAGE_NAME):latest
DOCKER_RUN=docker run $(RUN_FLAGS) $(IMAGE)

# make file commands
build:
	DOCKER_BUILDKIT=1 docker build --tag $(IMAGE) .

run:
	$(DOCKER_RUN) python $(example)

bash:
	$(DOCKER_RUN) bash
