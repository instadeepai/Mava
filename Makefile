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
USE_CUDA = $(if $(GPUS),true,false)

# make file commands
.PHONY: build
build:
	DOCKER_BUILDKIT=1 docker build --build-arg USE_CUDA=$(USE_CUDA) --tag $(IMAGE) .

.PHONY: run
run:
	$(DOCKER_RUN) python $(example)

.PHONY: bash
bash:
	$(DOCKER_RUN) bash

.PHONY: tpu_setup_conda
tpu_setup_conda:
	@wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
	bash miniconda.sh -b -p $$HOME/miniconda && \
	rm miniconda.sh && \
	export PATH="$$HOME/miniconda/bin:$$PATH" && \
	$$HOME/miniconda/bin/conda init && \
	$$HOME/miniconda/bin/conda create --name mava python=3.9 -y && \
	$$HOME/miniconda/envs/mava/bin/pip install -e .[dev] && \
	$$HOME/miniconda/envs/mava/bin/pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
