##########################################################
# Core Mava image
FROM nvidia/cuda:11.4.2-cudnn8-runtime-ubuntu20.04 as mava-core
# Flag to record agents
ARG record
# Ensure no installs try launch interactive screen
ARG DEBIAN_FRONTEND=noninteractive
# Update packages
RUN apt-get update --fix-missing -y && apt-get install -y python3-pip && apt-get install -y python3-venv
# Update python path
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 10 &&\
    rm -rf /root/.cache && apt-get clean
# Setup virtual env
RUN python -m venv mava
ENV VIRTUAL_ENV /mava
ENV PATH /mava/bin:$PATH
RUN pip install --upgrade pip setuptools wheel
# Location of mava folder
ARG folder=/home/app/mava
## working directory
WORKDIR ${folder}
## Copy code from current path.
COPY . /home/app/mava
# For box2d
RUN apt-get install swig -y
## Install core dependencies.
RUN pip install -e .[reverb,launchpad]
## Optional install for screen recording.
ENV DISPLAY=:0
RUN if [ "$record" = "true" ]; then \
    ./bash_scripts/install_record.sh; \
fi
EXPOSE 6006
##########################################################

##########################################################
# Core Mava-TF image
FROM mava-core as tf-core
# Tensorflow gpu config.
ENV TF_FORCE_GPU_ALLOW_GROWTH=true
ENV CUDA_DEVICE_ORDER=PCI_BUS_ID
ENV TF_CPP_MIN_LOG_LEVEL=3
## Install core tf dependencies.
RUN pip install -e .[tf]
##########################################################

##########################################################
# PZ image
FROM tf-core AS pz
RUN pip install -e .[pz]
# PettingZoo Atari envs
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get install -y unrar-free
RUN pip install autorom
RUN AutoROM -v
##########################################################

##########################################################
# SMAC image
FROM tf-core AS sc2
## Install smac environment
RUN apt-get -y install git
RUN pip install .[sc2]
# We use the pz wrapper for smac
RUN pip install .[pz]
ENV SC2PATH /home/app/mava/3rdparty/StarCraftII
##########################################################

##########################################################
# Flatland Image
FROM tf-core AS flatland
RUN pip install -e .[flatland]
##########################################################

#########################################################
## Robocup Image
FROM tf-core AS robocup
RUN apt-get install sudo -y
RUN ./bash_scripts/install_robocup.sh
##########################################################

##########################################################
## OpenSpiel Image
FROM tf-core AS openspiel
RUN pip install .[open_spiel]
##########################################################

##########################################################
# MeltingPot Image
FROM tf-core AS meltingpot
# Install meltingpot
RUN apt-get install -y git
RUN ./bash_scripts/install_meltingpot.sh
# Add meltingpot to python path
ENV PYTHONPATH "${PYTHONPATH}:${folder}/../packages/meltingpot"
##########################################################
