# Use slim Python base image
FROM python:3.10-slim

# Environment setup
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libfftw3-dev \
    libcfitsio-dev \
    libboost-all-dev \
    libeigen3-dev \
    ca-certificates \
    wget \
    curl \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Install shadems from PyPI
RUN pip install --upgrade pip && \
    pip install shadems==0.5.4 

# Default command
CMD ["shadems", "--help"]

