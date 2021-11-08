FROM tensorflow/tensorflow:2.5.0-gpu

RUN apt-get -y --fix-missing update

# working directory
WORKDIR /home/app/mava

# Tensorflow
ENV TF_FORCE_GPU_ALLOW_GROWTH=true
ENV CUDA_DEVICE_ORDER=PCI_BUS_ID
ENV TF_CPP_MIN_LOG_LEVEL=3

# PettingZoo
RUN apt-get install unrar
RUN python -m pip install autorom
RUN AutoROM -v

# cmake and clang for openspiel
RUN apt-get install clang -y
RUN python -m pip install cmake

# OpenCV
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

# Install `xvfb` to run a headless screen.
RUN apt-get update -y && \
    apt-get install -y xvfb && \
    apt-get install -y python-opengl
ENV DISPLAY=:0

# Install starcraft 2 environment
RUN apt-get -y install git
RUN pip install pysc2
RUN python -m pip uninstall -y enum34
RUN python -m pip install git+https://github.com/oxwhirl/smac.git
ENV SC2PATH /home/app/mava/3rdparty/StarCraftII

# Install Mava and dependencies
COPY . /home/app/mava
RUN python -m pip uninstall -y enum34
RUN python -m pip install --upgrade pip

# pyparsing is required as a prerequisite to the flatland install.
# The actual package installation order does not seem to correlate
# with the order of packages in flatland_requirements (system.py).
# Therefore the package is manually installed here.
RUN pip install pyparsing==3.0.3
RUN python -m pip install -e .[flatland]
RUN python -m pip install -e .[open_spiel]
RUN python -m pip install -e .[tf,envs,reverb,launchpad,testing_formatting,record_episode]

EXPOSE 6006
