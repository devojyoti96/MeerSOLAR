FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /opt

# Install dependencies (including bash)
RUN apt-get update && apt-get install -y \
    bash \
    build-essential \
    cmake \
    g++ \
    git \
    pkg-config \
    wget \
    curl \
    libblas-dev \
    libfftw3-dev \
    libboost-all-dev \
    libreadline-dev \
    libhdf5-dev \
    libpng-dev \
    casacore-dev \
    libgsl-dev \
    libboost-date-time-dev libboost-filesystem-dev \
    libboost-program-options-dev libboost-system-dev \
    libcfitsio-dev libfftw3-dev libgsl-dev \
    libhdf5-dev liblapack-dev libopenmpi-dev \
    libpython3-dev \
    vim \
    nano \
    && rm -rf /var/lib/apt/lists/*

# Clone and build WSClean v3.3
RUN git clone https://gitlab.com/aroffringa/wsclean.git && \
    cd wsclean && \
    git checkout v3.3 && \
    git submodule update --init --recursive
    
COPY ./chgcentre.cpp /opt/wsclean/chgcentre/main.cpp

RUN cd /opt/wsclean && \
    mkdir build && cd build && \
    cmake .. -DWSCLEAN_MP=ON -DWSCLEAN_BUILD_EVERYBEAM=ON && \
    make VERBOSE=1 -j"$(nproc)" && \
    make install && \
    cd /opt && rm -rf wsclean

# Ensure /usr/local/bin is in PATH
ENV PATH="/usr/local/bin:$PATH"

# Set entrypoint to bash for interactive or udocker use
CMD ["/bin/bash"]

