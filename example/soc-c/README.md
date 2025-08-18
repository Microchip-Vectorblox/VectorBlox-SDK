
# PolarFire SoC Video Kit (w/ VectorBlox V1000)

## Pre-requisites
- PolarFire SoC Video Kit, running latest [VectorBlox Demo design](https://github.com/Microchip-Vectorblox/VectorBlox-SoC-Video-Kit-Demo/releases), which uses the [Yocto 2023.02.1](https://github.com/polarfire-soc/meta-polarfire-soc-yocto-bsp/releases/download/v2023.02.1/core-image-minimal-dev-mpfs-video-kit-20230328105837.rootfs.wic.gz)
- Logged into Yocto as `root` with ethernet connection
- Upload `.vnnx` networks generated on host PC 
- Or alternatively, in the `/home/root` directory, download the quick start script:
    ```
    wget --no-check-certificate https://raw.githubusercontent.com/Microchip-Vectorblox/assets/refs/heads/main/quick_start.sh
    ```
- Run the script with the realease version you want to download. The command below shows how to download the `2.0.2` release (if no version is specified it will download the latest):
    ```
    bash quick_start.sh 2.0.2
    ```
    The `quick_start.sh` script will download the `VectorBlox-SDK-release-v2.0.2`, the `samples_V1000_2.0.2` sample networks, the `camera_setup.zip` camera setup and unzip them.
    
    Before continuing, make sure that all the files have been downloaded properly and unzipped succesfully.
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
