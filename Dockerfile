FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 as mava-core

# Ensure no installs try launch interactive screen
ARG DEBIAN_FRONTEND=noninteractive
# Update packages
RUN apt-get update -y && \
    apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt install -y python3.9 && \
    apt install -y python3.9-dev && \
    apt-get install -y python3-pip && \
    apt-get install -y python3.9-venv

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.9 10

# Check python v
RUN python -V

# Setup virtual env
RUN python -m venv mava
ENV VIRTUAL_ENV /mava
ENV PATH /mava/bin:$PATH
RUN pip install --upgrade pip setuptools wheel
# Location of mava folder
ARG folder=/home/app/mava
## working directory
WORKDIR ${folder}
## Copy code from current path
COPY . /home/app/mava
## Install core dependencies
RUN pip install -e .[dev]
RUN pip install --upgrade "jax[cuda11_pip]<=0.4.13" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

EXPOSE 6006
