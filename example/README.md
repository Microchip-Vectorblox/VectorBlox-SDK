# Simulator example
This directory contains a native executable to run networks as well as a series of python scripts to run networks.

In order to run these examples you must have created a model blob. See documentation else where on how
to create that model blob.

## Python scripts

The python scripts located in examples/python/ can be run after the vbx python module is installed. Run with the --help
argument to discover the correct usage of the scripts.

## Native Executable

### Prebuilt program


The program is called `sim-run-model` and has one required argument - the model - and one optional argument - an image.

### Source

To build the source run the following command from the examples/c/ directory

```
bash build.sh
```
