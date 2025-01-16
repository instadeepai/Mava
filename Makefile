# Check if GPU is available - if `nvidia-smi` works then use GPUs
GPUS := $(shell command -v nvidia-smi > /dev/null && nvidia-smi > /dev/null 2>&1 && echo "--gpus all" || echo "")

# Set flag for docker run command
BASE_FLAGS=-it --rm
RUN_FLAGS=$(GPUS) $(BASE_FLAGS)

DOCKER_IMAGE_NAME = mava
IMAGE = $(DOCKER_IMAGE_NAME):latest
DOCKER_RUN=docker run $(RUN_FLAGS)
USE_CUDA = $(if $(GPUS),true,false)

# Mount path for mava folder
MOUNT_FLAGS=-v $(PWD)/:/home/app/mava

# make file commands
build:
	DOCKER_BUILDKIT=1 docker build --build-arg USE_CUDA=$(USE_CUDA) --tag $(IMAGE) .

run:
	$(DOCKER_RUN) $(IMAGE) python $(example)

run_mount:
	$(DOCKER_RUN) $(MOUNT_FLAGS) -e NEPTUNE_API_TOKEN=$(NEPTUNE_API_TOKEN) $(IMAGE) python $(example)

bash:
	$(DOCKER_RUN) $(MOUNT_FLAGS) $(IMAGE) bash
