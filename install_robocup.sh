sudo apt-get update \
    && apt-get install -y --no-install-recommends \
       apt-utils \
       build-essential \
       curl \
       xvfb \
       ffmpeg \
       xorg-dev \
       libsdl2-dev \
       swig \
       cmake \
       python-opengl

# fetch repo / ppa packages, etc
sudo apt-get -y update --fix-missing
# Install package, that hangs the operation, separately
sudo DEBIAN_FRONTEND=noninteractive apt install -y tzdata

sudo apt update && \
    apt -y install autoconf bison clang flex libboost-dev libboost-all-dev libc6-dev make wget

sudo apt -y install build-essential libboost-all-dev qt5-default libfontconfig1-dev libaudio-dev libxt-dev libglib2.0-dev libxi-dev libxrender-dev libboost-all-dev

sudo wget https://github.com/rcsoccersim/rcssserver/archive/rcssserver-16.0.0.tar.gz && \
    tar xfz rcssserver-16.0.0.tar.gz && \
    cd rcssserver-rcssserver-16.0.0 && \
    ./bootstrap && \
    ./configure && \
    make && \
    make install && \
    ldconfig

sudo wget https://github.com/rcsoccersim/rcssmonitor/archive/rcssmonitor-16.0.0.tar.gz && \
    tar xfz rcssmonitor-16.0.0.tar.gz && \
    cd rcssmonitor-rcssmonitor-16.0.0 && \
    ./bootstrap && \
    ./configure && \
    make && \
    make install && \
    ldconfig

sudo ldconfig && \
    apt update && \
    apt install -y libboost-filesystem1.65.1 libboost-system1.65.1 libboost-program-options-dev tmux

sudo apt-get install -y libqt5widgets5
# RUN pip install hydra-core

# ENV PYTHONPATH '/home'
