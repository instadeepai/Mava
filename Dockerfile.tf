


# Mava core tf image
FROM tensorflow/tensorflow:2.7.0-gpu as tf-core
# Flag to record agents
ARG record
RUN apt-get -y --fix-missing update
## working directory
WORKDIR /home/app/mava
# Tensorflow gpu config.
ENV TF_FORCE_GPU_ALLOW_GROWTH=true
ENV CUDA_DEVICE_ORDER=PCI_BUS_ID
ENV TF_CPP_MIN_LOG_LEVEL=3
## Copy code from current path.
COPY . /home/app/mava
RUN python -m pip uninstall -y enum34
RUN python -m pip install --upgrade pip
# For box2d
RUN apt-get install swig -y
## Install core dependencies.
RUN python -m pip install -e .[tf,reverb,launchpad]
## Optional install for screen recording.
ENV DISPLAY=:0
RUN if [ "$record" = "true" ]; then \
    ./bash_scripts/install_record.sh; \
fi
EXPOSE 6006
##########################################################

##########################################################
# PZ image
FROM tf-core AS pz
RUN python -m pip install -e .[pz]
# PettingZoo Atari envs
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get install unrar-free
RUN python -m pip install autorom
RUN AutoROM -v
##########################################################

##########################################################
# SMAC image
FROM tf-core AS sc2
## Install SC2 game
RUN apt-get install -y wget
## Install smac environment
RUN apt-get -y install git
RUN pip install .[sc2]
RUN ./bash_scripts/install_sc2.sh
ENV SC2PATH /home/app/mava/3rdparty/StarCraftII
##########################################################

##########################################################
# Flatland Image
FROM tf-core AS flatland
RUN python -m pip install -e .[flatland]
##########################################################

##########################################################
## Robocup Image
FROM tf-core AS robocup
RUN ./bash_scripts/install_robocup.sh
##########################################################

##########################################################
## OpenSpiel Image
FROM tf-core AS openspiel
RUN pip install .[open_spiel]
##########################################################
