#!/bin/bash
# Copyright (C) 2021 Microchip Technologies


# install dependencies
if grep -iEq "ubuntu 20.04|ubuntu 22.04|ubuntu 24.04" /etc/issue
then
    sudo apt update
    sudo DEBIAN_FRONTEND=noninteractive apt install -y \
	    software-properties-common build-essential ninja-build\
	    curl wget unzip \
	    git git-lfs \
	    cmake protobuf-compiler \
	    libenchant-2-dev libopenblas-dev libgl-dev libusb-1.0-0-dev libjpeg-dev

    sudo add-apt-repository ppa:deadsnakes/ppa -y
    sudo apt update
    sudo apt install -y \
	    python3.10-dev python3.10-venv \

    wget https://github.com/PINTO0309/onnx2tf/releases/download/1.16.31/flatc.tar.gz \
        && tar -zxvf flatc.tar.gz \
        && sudo chmod +x flatc \
        && sudo mv flatc /usr/bin/
else
    echo "Unsupported OS, please install dependencies manually" && exit 1
fi
