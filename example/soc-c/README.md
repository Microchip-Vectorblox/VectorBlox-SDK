
# PolarFire SoC Video Kit (w/ VectorBlox V1000)

## Pre-requisites
- PolarFire SoC Video Kit, running latest [VectorBlox Demo design](https://github.com/Microchip-Vectorblox/VectorBlox-SoC-Video-Kit-Demo/releases), which uses the [Yocto 2023.02.1](https://github.com/polarfire-soc/meta-polarfire-soc-yocto-bsp/releases/download/v2023.02.1/core-image-minimal-dev-mpfs-video-kit-20230328105837.rootfs.wic.gz)
- Logged into Yocto as `root` with ethernet connection
- Upload `.vnnx` networks generated on host PC 
- Or alternatively, download and unzip the [sample networks](https://github.com/Microchip-Vectorblox/VectorBlox-SoC-Video-Kit-Demo/releases/download/release-v2.0.2/samples_V1000_2.0.2.zip)
    ```
    wget --no-check-certificate  https://github.com/Microchip-Vectorblox/VectorBlox-SoC-Video-Kit-Demo/releases/download/release-v2.0.2/samples_V1000_2.0.2.zip 

    unzip samples_V1000_2.0.2.zip
    ```
- Download and unzip the [VectorBlox SDK](https://github.com/Microchip-Vectorblox/VectorBlox-SDK/archive/refs/tags/release-v2.0.2.zip), and navigate to this example

    ```
    wget --no-check-certificate https://github.com/Microchip-Vectorblox/VectorBlox-SDK/archive/refs/tags/release-v2.0.2.zip 

    unzip release-v2.0.2.zip
    cd VectorBlox-SDK-release-v2.0.2/example/soc-video-c
    ```
 ## Using `run-model` to benchmark networks
- Run `make overlay` to add the VectorBlox instance to the device tree (required every boot)
- Run `make` to build the demo application  
- Run `./run-model`  with the following arguments: `FIRMWARE.bin MODEL.vnnx IMAGE.jpg [POST_PROCESS]`
    - `TEST_DATA` can be specified to use a model's internal test data in place of an image to verify hardware and simulator bit-accuracy (via `CHECKSUM`)
    - `POST_PROCESS` modes supported: `CLASSIFY, YOLOV2, YOLOV3, YOLOV4, YOLOV5, BLAZEFACE, SCRFD, RETINAFACE, SSDV2, PLATE, LPD, LPR, ULTRALYTICS, ULTRALYTICS_FULL, ULTRALYTICS_PIXEL` (or left blank)
    
## Examples usage 
```
./run-model ~/samples_V1000_2.0.2/mobilenet-v2.vnnx ../../tutorials/test_images/oreo.jpg  CLASSIFY
./run-model ~/samples_V1000_2.0.2/mobilenet-v2.vnnx TEST_DATA 
```
