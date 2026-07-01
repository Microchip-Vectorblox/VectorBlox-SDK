# Examples

This directory contains examples that run `.vnnx` networks using the VectorBlox API. The Python scripts and `sim-c` use the simulator on the user's host PC, while the `soc-c` and `soc-video-c` examples build and run on a PolarFire SoC FPGA running Yocto Linux.

## Simulation Examples for SDK

- Python scripts are used to verify networks and are called in the various tutorials. The `VBX_SDK` Python environment must be installed before running.

  > Run a script with the `--help` argument to display usage.

- `sim-c` runs a `.vnnx` network using the simulator. Additional information [here](./sim-c).

## Hardware Reference Design Examples for PolarFire SoC

- `soc-c` runs a `.vnnx` network on the PolarFire SoC Video Kit. Additional information [here](./soc-c).
- `soc-video-c` runs a video demo on the PolarFire SoC Video Kit. Additional information [here](./soc-video-c).
