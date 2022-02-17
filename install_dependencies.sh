#!/bin/bash
# Copyright (C) 2021 Microchip Technologies


# install dependencies
if grep -iEq "ubuntu (16|18|20).04" /etc/issue
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
         python3-venv \
         python3-dev \
         build-essential   
    if grep -iEq "ubuntu 20.04" /etc/issue
    then
        #on 20.04 a few more packages are needed to build some pip wheels
        sudo apt-get install -y cmake protobuf-compiler libenchant-dev
    fi
else
    echo "Unknown OS, please install dependencies manually" && exit 1
fi
