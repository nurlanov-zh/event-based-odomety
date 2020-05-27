# Use the official image as a parent image.
FROM ubuntu:18.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential \
    gdb \
    cmake \
    git \
    libtbb-dev \
    libeigen3-dev \
    libglew-dev \
    ccache \
    libjpeg-dev \
    libpng-dev \
    openssh-client \
    liblz4-dev \
    libbz2-dev \
    libboost-regex-dev \
    libboost-filesystem-dev \
    libboost-date-time-dev \
    libboost-program-options-dev \
    libopencv-dev \
    libpython2.7-dev \
    gfortran \
    libc++-dev \
    libgoogle-glog-dev \
    libatlas-base-dev \
    libsuitesparse-dev \
    wget \
    valgrind
