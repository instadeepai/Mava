FROM tensorflow/tensorflow:latest-gpu

RUN apt-get -y --fix-missing update

# working directory
WORKDIR /home/app/mava

# Tensorflow
ENV TF_FORCE_GPU_ALLOW_GROWTH=true
ENV CUDA_DEVICE_ORDER=PCI_BUS_ID
ENV TF_CPP_MIN_LOG_LEVEL=3

# Install Mava and dependencies
COPY . /home/app/mava
RUN python -m pip install --upgrade pip
RUN python -m pip install -e .[tf,envs,reverb,launchpad,testing_formatting]

EXPOSE 6006
