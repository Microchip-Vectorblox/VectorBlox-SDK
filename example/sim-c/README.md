## Pre-requisites
- Setup `VBX_SDK` environment (see steps  [here](../../README.md))
- Generate a vnnx network from the [tutorials directory](../../tutorials/)

- Or alternatively, [sample networks]( https://github.com/Microchip-Vectorblox/VectorBlox-SoC-Video-Kit-Demo/releases/download/release-v2.0.3/samples_V1000_2.0.3.zip) can be downloaded directly 
    ```
    wget --no-check-certificate https://github.com/Microchip-Vectorblox/VectorBlox-SoC-Video-Kit-Demo/releases/download/release-v2.0.3/samples_V1000_2.0.3.zip
 
    unzip samples_V1000_2.0.3.zip
    ```
## Using  `sim-run-model` to benchmark networks
- Activate the `VBX_SDK` environment
- Run `make` to build the demo application
- Run `./sim-run-model`  with the following arguments: `MODEL.vnnx IMAGE.jpg [POST_PROCESS]`
     - `TEST_DATA` can be specified to use a model's internal test data in place of an image to verify hardware and simulator bit-accuracy (via `CHECKSUM`)
    - Current values supported for `POST_PROCESS` are the following: `CLASSIFY, YOLOV2, YOLOV3, YOLOV4, YOLOV5, BLAZEFACE, SCRFD, RETINAFACE, SSDV2, PLATE, LPD, LPR, ULTRALYTICS, ULTRALYTICS_FULL, ULTRALYTICS_POSE` (or left blank)
    
## Examples usage 
```
./sim-run-model  ~/samples_V1000_2.0.3/mobilenet-v2.vnnx ../../tutorials/test_images/oreo.jpg CLASSIFY
./sim-run-model  ~/samples_V1000_2.0.3/mobilenet-v2.vnnx ../../tutorials/test_images/oreo.jpg  
./sim-run-model  ~/samples_V1000_2.0.3/mobilenet-v2.vnnx TEST_DATA
```
    

