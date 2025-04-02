# VectorBlox SDK

VectorBlox is an SDK for compiling and running TFLITE INT8 networks on the VectorBlox accelerator.

The list of the TFLITE INT8 supported operators can be found [here](./docs/OPS.md) (currently 64 supported).

Networks must be quantized, then compiled via the included scripts.
The networks can then be run via the VectorBlox simulator, or on a physical VectorBlox accelerator.

## Prerequisites


 To use the VectorBlox SDK, you will need run the SDK in an Ubuntu 20.04, 22.04, or 24.04 environment.
 If using WSL, ensure you are running from your home directory or have permissions set.

> If cloning the repo, `git` and `git-lfs` must be installed via `apt install git-lfs && git-lfs install`

## Downloading the SDK

There are two options for downloading the sdk:

 1) Download an archive (zip or tar.gz) from https://github.com/Microchip-Vectorblox/VectorBlox-SDK/releases
     
 2) Clone this repository
     
### Install dependencies (done once, requires sudo permission)

Navigate to the root directory of `VectorBlox-SDK` and run the following commands:

```
bash install_dependencies.sh
```

### Activate (and installs if needed) Python3.10 virtual environment, and sets necessary environment variables
```
source setup_vars.sh
```

## Run tutorials

Example tutorials are provided that generate CoreVectorBlox compatible Binary Large OBjects (BLOBs).

Tutorials will download the model file and convert it to a quantized `.tflite` if necessary.
It will then be compiled into a Vectorblox BLOB (`.vnnx` and `.hex` extensions) and simulated.

Follow the below steps to run a tutorial.

```
cd $VBX_SDK/tutorials/{network_source}/{network}
bash {network}.sh
```

A list of tutorials can be found [here](./tutorials/README.md)
> If experiencing CUDA issues, with the `vbx_env` active, install the CPU version of TensorFlow via: `pip uninstall tensorflow && pip install tensorflow-cpu==2.15.1`
