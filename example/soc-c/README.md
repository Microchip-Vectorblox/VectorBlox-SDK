# PolarFire SoC Video Kit (with VectorBlox V1000)

This is an I/O-based command-line application for the [VectorBlox SoC Demo Design](https://github.com/Microchip-Vectorblox/VectorBlox-SoC-Video-Kit-Demo). The application expects two arguments: a model in VNNX format and an image file, and outputs latency and inference results to a UART terminal.

## Pre-requisites

- Follow the steps on [VectorBlox SoC Demo Design](https://github.com/Microchip-Vectorblox/VectorBlox-SoC-Video-Kit-Demo), including the step to build the demo.

## Using `run-model` to benchmark networks

- Run `make overlay` to add the VectorBlox instance to the device tree (required every boot). If `make overlay` reports error `Overlay mpfs_vbx.dtbo already exists`, this can be ignored.
- Run `make` to build the demo application.
- Run `./run-model` with the following arguments: `MODEL.vnnx IMAGE.jpg [POST_PROCESS]`
  - `TEST_DATA` can be specified in place of an image to use a model's internal test data and verify hardware and simulator bit-accuracy (via `CHECKSUM`).
  - Current values supported for `POST_PROCESS` can be found in the post-processing documentation [here](../../docs/C_Postprocessing.md).

## Example usage

```bash
./run-model mobilenet-v2.vnnx ../../tutorials/test_images/oreo.jpg CLASSIFY
./run-model mobilenet-v2.vnnx ../../tutorials/test_images/oreo.jpg
./run-model mobilenet-v2.vnnx TEST_DATA
```
