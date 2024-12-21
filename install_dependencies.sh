#!/bin/bash
# Copyright (C) 2021 Microchip Technologies


# install dependencies
if grep -iEq "ubuntu 20.04" /etc/issue
then
    # Ubuntu
    sudo apt update
  	sudo DEBIAN_FRONTEND=noninteractive apt-get install -y libopenblas-base \
         git \
         libsm6 \
         libxrender1 \
	 libgl1 \
         curl \
         wget \
         unzip \
         python3-enchant \
         python3.9-venv \
         python3.9-dev \
         libjpeg-dev \
         build-essential   
    sudo apt-get install -y cmake protobuf-compiler libenchant-dev libjpeg-dev
    wget https://github.com/PINTO0309/onnx2tf/releases/download/1.16.31/flatc.tar.gz \
        && tar -zxvf flatc.tar.gz \
        && sudo chmod +x flatc \
        && sudo mv flatc /usr/bin/
else
    echo "Unsupported OS, please install dependencies manually" && exit 1
fi
