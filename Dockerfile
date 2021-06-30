FROM tensorflow/tensorflow:2.5.0-gpu

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

# OpenCV
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

# Install `xvfb` to run a headless screen.
RUN apt-get update -y && \
    apt-get install -y xvfb && \
    apt-get install -y python-opengl
ENV DISPLAY=:0

RUN export PIP_DEFAULT_TIMEOUT=100
# Ensure encoding consistency
RUN export LC_ALL=en_US.UTF-8
RUN export LANG=en_US.UTF-8

# Mava dependencies
COPY . /home/app/mava
RUN apt-get install build-essential
RUN python -m pip uninstall -y enum34
RUN python -m pip install --upgrade pip setuptools
RUN python -m pip install .
RUN python -m pip install .[flatland]

# Openspiel
RUN apt-get update
RUN apt-get install clang -y
RUN python -m pip install .[open_spiel]
# Other Mava dependencies
RUN python -m pip install .[tf,envs,reverb,testing_formatting,launchpad,record_episode]
# For atari envs
RUN apt-get install unrar
RUN python -m pip install autorom
RUN AutoROM -v
# Open CV and Headless screen.
RUN apt-get install ffmpeg libsm6 libxext6 xvfb  -y

# Install starcraft 2 environment
RUN apt-get -y install git
RUN python -m pip install git+https://github.com/oxwhirl/smac.git
ENV SC2PATH /home/app/mava/3rdparty/StarCraftII
EXPOSE 6006
