
# PolarFire SoC Video Kit (w/ VectorBlox V1000)

## Pre-requisites

- Follow steps on [VectorBlox SoC Demo Design](https://github.com/Microchip-Vectorblox/VectorBlox-SoC-Video-Kit-Demo), including the step to build the demo.

 ## Using `run-model` to benchmark networks
- Run `make overlay` to add the VectorBlox instance to the device tree (required every boot)
- Run `make` to build the demo application  
- Run `./run-model`  with the following arguments: `MODEL.vnnx IMAGE.jpg [POST_PROCESS]`
    - `TEST_DATA` can be specified to use a model's internal test data in place of an image to verify hardware and simulator bit-accuracy (via `CHECKSUM`)
    - Current values supported for `POST_PROCESS` can be found in the postprocess documentation [here](../../docs/postprocess.md)
    
## Examples usage 
```
./run-model mobilenet-v2.vnnx ../../tutorials/test_images/oreo.jpg CLASSIFY
./run-model mobilenet-v2.vnnx ../../tutorials/test_images/oreo.jpg
./run-model mobilenet-v2.vnnx TEST_DATA 
```
