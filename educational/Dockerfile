# Mava Edu image
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 as mava-edu
# Flag to record agents
ARG record
# Ensure no installs try launch interactive screen
ARG DEBIAN_FRONTEND=noninteractive
# Update packages
RUN apt-get update -y && \
    apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt install -y python3.9 && \
    apt install -y python3.9-dev && \
    apt-get install -y python3-pip && \
    apt-get install -y python3.9-venv && \ 
    apt-get -y install git

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.9 10

# Check python v
RUN python -V

# Jax Config
# Jax gpu config.
ENV XLA_PYTHON_CLIENT_PREALLOCATE=false

# PettingZoo Atari envs
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get install -y unrar-free
RUN pip install autorom
RUN AutoROM -v

# SC2 Path
ENV SC2PATH /home/app/mava/3rdparty/StarCraftII

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
# For box2d
RUN apt-get install swig -y

# Install packages
RUN pip install -r requirements.txt 
# Install jax gpu
RUN pip install --upgrade "jax[cuda]==0.4.1" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
## Optional install for screen recording.
ENV DISPLAY=:0
RUN if [ "$record" = "true" ]; then \
    ./bash_scripts/install_record.sh; \
    fi
EXPOSE 6006