# Examples
This directory contains 3 examples of running networks using the Vectorblox API.

1. A C executable for running networks on the simulator
2. A Python script for running networks on the simulator
3. A C executable for running networks on PFSoC (BETA)


## Simulator C

The exectuable can be built by running `build.sh`. The command takes 3 arguments. The first argument is the network to run.
The second is the jpeg file to use as the input to the network. The third is the type of post processing 
(CLASSIFY,YOLOV2,TINYYOLOV2,TINYYOLOV3)

## SoC C

Undocumented, in beta


## Python scripts

The python scripts located in examples/python/ can be run after the vbx python module is installed. Run with the --help
argument to discover the correct usage of the scripts.


```
