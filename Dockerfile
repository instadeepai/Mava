FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Ensure no installs try to launch interactive screen
ARG DEBIAN_FRONTEND=noninteractive

# Update packages and install python3.9 and other dependencies
RUN apt-get update -y && \
    apt-get install -y software-properties-common git && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get install -y python3.12 python3.12-dev python3-pip python3.12-venv && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.12 10 && \
    python -m venv mava && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Setup virtual env and path
ENV VIRTUAL_ENV /mava
ENV PATH /mava/bin:$PATH

# Location of mava folder
ARG folder=/home/app/mava

# Set working directory
WORKDIR ${folder}

# Copy all code needed to install dependencies
COPY ./requirements ./requirements
COPY setup.py .
COPY README.md .
COPY mava/version.py mava/version.py

RUN echo "Installing requirements..."
RUN pip install --quiet --upgrade pip setuptools wheel &&  \
    pip install -e .

# Need to use specific cuda versions for jax
ARG USE_CUDA=true
RUN if [ "$USE_CUDA" = true ] ; \
    then pip install "jax[cuda12]==0.4.25" ; \
    fi

# Copy all code
COPY . .

EXPOSE 6006
