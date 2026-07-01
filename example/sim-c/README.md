# VectorBlox SDK C Simulation

This directory is for simulating and benchmarking networks using the C API, as if they were running on hardware.

## Pre-requisites

- Set up the `VBX_SDK` environment (see steps [here](../../README.md)).
- Generate a `.vnnx` network from the [tutorials directory](../../tutorials/).

## Using `sim-run-model` to benchmark networks

- Activate the `VBX_SDK` environment.
- Run `make` to build the demo application.
- Run `./sim-run-model` with the following arguments: `MODEL.vnnx IMAGE.jpg [POST_PROCESS]`
  - `TEST_DATA` can be specified in place of an image to use a model's internal test data and verify hardware and simulator bit-accuracy (via `CHECKSUM`).
  - Current values supported for `POST_PROCESS` can be found in the post-processing documentation [here](../../docs/C_Postprocessing.md).

## Example usage

```bash
./sim-run-model mobilenet-v2.vnnx ../../tutorials/test_images/oreo.jpg CLASSIFY
./sim-run-model mobilenet-v2.vnnx ../../tutorials/test_images/oreo.jpg
./sim-run-model mobilenet-v2.vnnx TEST_DATA
```
