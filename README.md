# Prerequisites

> Ubuntu 16.04 / 18.04 supported
> If using WSL, ensure you are running from your home directory or have permissions set

## Installs required dependencies (requires sudo permission)

```
bash install_dependencies.sh
```

## Activates (and installs if needed) Python virtual environment, and sets necessary environment variables
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


