# Prerequisites

 Ubuntu 16.04 / 18.04 supported
 If using WSL, ensure you are running from your home directory or have permissions set

# Downloading the SDK

There are two options for downloading the sdk:

 1) Download an archive.  
  Download the zip or tar.gz from https://github.com/Microchip-Vectorblox/VectorBlox-SDK/releases
     
 2) Git clone  
    In order to clone it is necessary to have [git-lfs](https://git-lfs.github.com/) installed.To install git-lfs run the following commands on ubuntu:
    ```
    curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
    apt update && apt install git-lfs
    git lfs install
    ```
    If during `source setup_vars.sh` you get an error about "No module named 'mxnet'"
    chances are git-lfs is not configured correctly.
     
## Install required dependencies (requires sudo permission)

```
bash install_dependencies.sh
```

## Activate (and installs if needed) Python virtual environment, and sets necessary environment variables
```
source setup_vars.sh
```

# Run tutorials

Several tutorials are provide to generate CoreVectoBlox compatible Binary Large OBjects (BLOBs)

The tutorials download the networks, convert and optimize the network into openvino xml, and
then finally convert the the xml into a quantized Vectoblox BLOB with a .vnnx or .hex file extension.

To run the tutorials, follow the below steps. 

```
cd tutorials/{framework}/{network}
bash {network}.sh
```

A list of tutorials can be found [here](./tutorials/README.md)


