#!/bin/bash
# Copyright (C) 2018 Intel Corporation


# install dependencies
if grep -iEq "ubuntu (16|18|20).04" /etc/issue
then
    # Ubuntu
	export DEBIAN_FRONTEND=noninteractive
    sudo -E apt update
    sudo -E apt-get install -y \
  		 libopenblas-dev \
  		 git \
		 libsm6 \
		 libxext6 \
		 libxrender1 \
		 curl \
		 wget \
		 unzip \
		 python3-enchant \
	     libtbb2 \
		 libomp-dev \
		 python3-venv \
		 python3-dev \
		 build-essential \
		 aria2
	
else
    echo "Unknown OS, please install dependencies manually" && exit 1
fi
