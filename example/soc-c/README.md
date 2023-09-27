
# PolarFire SoC Video Kit (w/ VectorBlox V1000)

## Pre-requisites
- PolarFire SoC Video Kit, running latest [VectorBlox Demo design](https://github.com/Microchip-Vectorblox/VectorBlox-SoC-Video-Kit-Demo/releases), which uses the [Yocto 2023.02.1](https://github.com/polarfire-soc/meta-polarfire-soc-yocto-bsp/releases/download/v2023.02.1/core-image-minimal-dev-mpfs-video-kit-20230328105837.rootfs.wic.gz)
- Logged into Yocto as `root` with ethernet connection
- Upload `.vnnx` networks generated on host PC 
- Or alternatively, download and unzip the [sample networks](https://vector-blox-model-zoo.s3.us-west-2.amazonaws.com/Releases/ModelZoo/samples_V1000_1.4.4.zip)
    ```
    wget --no-check-certificate https://vector-blox-model-zoo.s3.us-west-2.amazonaws.com/Releases/ModelZoo/samples_V1000_1.4.4.zip 

    unzip samples_V1000_1.4.4.zip
    ```
- Download and unzip the [VectorBlox SDK](https://github.com/Microchip-Vectorblox/VectorBlox-SDK/archive/refs/tags/release-v1.4.4.1.zip), and navigate to this example

    ```
    wget --no-check-certificate https://github.com/Microchip-Vectorblox/VectorBlox-SDK/archive/refs/tags/release-v1.4.4.1.zip 

    unzip release-v1.4.4.1.zip 
    cd VectorBlox-SDK-release-v1.4.4.1/example/soc-c
    ```
 ## Using `run-model` to benchmark networks
- Run `make overlay` to add the VectorBlox instance to the device tree (required every boot)
- Run `make` to build the demo application  
- Run `./run-model`  with the following arguments: `FIRMWARE.bin MODEL.vnnx IMAGE.jpg [POST_PROCESS]`
    - `TEST_DATA` can be specified to use a model's internal test data in place of an image to verify hardware and simulator bit-accuracy (via `CHECKSUM`)
    - `POST_PROCESS` modes supported: `CLASSIFY, YOLOV2, YOLOV3, YOLOV4, YOLOV5, BLAZEFACE, SCRFD, RETINAFACE, SSDV2, PLATE, LPD, LPR` (or left blank)
    
## Examples usage 
```
./run-model ../../fw/firmware.bin ~/samples_V1000_1.4.4/mobilenet-v2.vnnx ../../tutorials/test_images/oreo.jpg  CLASSIFY
./run-model ../../fw/firmware.bin ~/samples_V1000_1.4.4/mobilenet-v2.vnnx TEST_DATA 
```


# PolarFire SoC Icicle Kit (w/ VectorBlox V500)

## Pre-requisites
- PolarFire SoC Icicle Kit running [VectorBlox design](https://github.com/polarfire-soc/icicle-kit-reference-design/releases/download/v2023.06/MPFS_ICICLE_VECTORBLOX_2023_06.zip), which uses the [Yocto 2023.06](https://github.com/polarfire-soc/meta-polarfire-soc-yocto-bsp/releases/download/v2023.06/core-image-minimal-dev-icicle-kit-es-20230629084406.rootfs.wic.gz)
  > If having trouble booting Yocto Linux, updated HSS by running latest [reference design](https://github.com/polarfire-soc/icicle-kit-reference-design/releases/download/v2023.06/MPFS_ICICLE_BASE_DESIGN_2023_06.zip), then rerun VectorBlox design
- Logged into Yocto as `root` with ethernet connection
- Upload `.vnnx` networks generated on host PC 
- Or alternatively, download and unzip the [sample networks](https://vector-blox-model-zoo.s3.us-west-2.amazonaws.com/Releases/ModelZoo/samples_V500_1.4.4.zip)
    ```
    wget --no-check-certificate https://vector-blox-model-zoo.s3.us-west-2.amazonaws.com/Releases/ModelZoo/samples_V500_1.4.4.zip 

    unzip samples_V500_1.4.4.zip
    ```
- Download and unzip the [VectorBlox SDK](https://github.com/Microchip-Vectorblox/VectorBlox-SDK/archive/refs/tags/release-v1.4.4.1.zip), and navigate to this example

    ```
    wget --no-check-certificate https://github.com/Microchip-Vectorblox/VectorBlox-SDK/archive/refs/tags/release-v1.4.4.1.zip 

    unzip release-v1.4.4.1.zip 
    cd VectorBlox-SDK-release-v1.4.4.1/example/soc-c
    ```
 ## Using `run-model` to benchmark networks
- Run `make overlay` to add the VectorBlox instance to the device tree (required every boot)
- Run `make kit=icicle` to build the demo application  
    > The Icicle Kit uses the SoC DDR controller (`MSS_DDR=1`) and is not setup to use interrupts (`USE_INTERRUPTS=0`)   
    > On the current release of the Icicle Kit, models requiring >32 MB cannot be allocated and ran.
- Run `./run-model`  with the following arguments: `FIRMWARE.bin MODEL.vnnx IMAGE.jpg [POST_PROCESS]`
    - `TEST_DATA` can be specified to use a model's internal test data in place of an image to verify hardware and simulator bit-accuracy (via `CHECKSUM`)
    - `POST_PROCESS` modes supported: `CLASSIFY, YOLOV2, YOLOV3, YOLOV4, YOLOV5, BLAZEFACE, SCRFD, RETINAFACE, SSDV2, PLATE, LPD, LPR` (or left blank)
    
## Examples usage 
```
./run-model ../../fw/firmware.bin ~/samples_V500_1.4.4/mobilenet-v2.vnnx ../../tutorials/test_images/oreo.jpg  CLASSIFY
./run-model ../../fw/firmware.bin ~/samples_V500_1.4.4/mobilenet-v2.vnnx TEST_DATA 
```
