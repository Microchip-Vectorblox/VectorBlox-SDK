# Examples

This directory contains examples that run `.vnnx` networks using the VectorBlox API. The Python scripts and `sim-c` use the simulator on the user's host PC, while the latter examples build and run on a PFSoC FPGA, running Yocto Linux.

- Python scripts are used to verify networks and are called in the various tutorials. The `VBX_SDK` Python environment must be installed before running. 
 > Run a script with `--help` argument to display usage.
- `sim-c` runs a '.vnnx' network using the simulator. Additional information [here](./sim-c)
- `soc-c` runs a `.vnnx` network on the PFSoC Video Kit. Additional information [here](./soc-c)
- `soc-video-c` runs a video demo on the PFSoC Video Kit. Additional information [here](./soc-video-c)
