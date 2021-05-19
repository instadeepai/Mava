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


# OpenCV
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

# Install `xvfb` to run a headless screen.
RUN apt-get update -y && \
    apt-get install -y xvfb && \
    apt-get install -y python-opengl
ENV DISPLAY=:0

# Install Mava and dependencies
COPY . /home/app/mava
RUN python -m pip install --upgrade pip
RUN python -m pip install -e .[flatland]
RUN python -m pip install -e .[tf,envs,reverb,launchpad,testing_formatting,record_episode]

EXPOSE 6006
