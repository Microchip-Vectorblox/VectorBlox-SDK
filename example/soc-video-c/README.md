## Pre-requisites

- PolarFire SoC Video Kit, running latest [VectorBlox Demo design](https://github.com/Microchip-Vectorblox/VectorBlox-SoC-Video-Kit-Demo/releases), which uses [Yocto 2023.02.1](https://github.com/polarfire-soc/meta-polarfire-soc-yocto-bsp/releases/download/v2023.02.1/core-image-minimal-dev-mpfs-video-kit-20230328105837.rootfs.wic.gz)
- Logged into Yocto as `root` with ethernet connection
- Download and unzip the [sample networks](https://vector-blox-model-zoo.s3.us-west-2.amazonaws.com/Releases/ModelZoo/samples_V1000_1.4.4.zip) to the root directory
    ```
    wget --no-check-certificate https://vector-blox-model-zoo.s3.us-west-2.amazonaws.com/Releases/ModelZoo/samples_V1000_1.4.4.zip 

    unzip samples_V1000_1.4.4.zip
    ```
- Download and unzip the [VectorBlox SDK](https://github.com/Microchip-Vectorblox/VectorBlox-SDK/archive/refs/tags/release-v1.4.4.1.zip), and navigate to this example

    ```
    wget --no-check-certificate https://github.com/Microchip-Vectorblox/VectorBlox-SDK/archive/refs/tags/release-v1.4.4.1.zip 

    unzip release-v1.4.4.1.zip
    cd release-v1.4.4.1.zip/example/soc-video-c
    ```
- HDMI cables connected to the PolarFire SoC Video Kit (Rx/Tx)
    > Users can run `make hdmi` to toggle the frame buffers (for ~1 sec) to verify their HDMI setup. Please check cables and power-cycle board if any issues.

## Starting the VectorBlox HDMI demo on the PolarFire SoC Video Kit

- Run `make overlay` to add the VectorBlox instance to the device tree (required every boot)
- Run `make` to build the demo application
- Run `./run-video-model` to launch the demo

## Controlling the VectorBlox HDMI demo on the PolarFire SoC Video Kit

- The demo consists of the following:
    - Face Recognition Demo
    - License Plate Demo
    - Classification (Mobilenetv2)
    - Object Detection (YOLOv5 Nano, YOLOv4 Tiny) 
- Use the `ENTER` key to switch modes. Entering `q` (pressing `q` and `ENTER`) quits the demo
- In the `Face Recognition` mode, you can enter `a` to add or `d` to delete face embeddings
    - Entering `a` initially highlights the largest face on-screen, entering `a` again adds that face to the embeddings. You will then be prompted to enter a name( or just press `ENTER` to use the default ID)

    - Entering `d` will list hte indices and names of the embeddings. Enter the desired index to delete the specified embedding from the database (or press `ENTER` to skip the deletion)


Samples videos for input to the Faces Recognition and License Plate Recognition modes are available [here](https://vector-blox-model-zoo.s3.us-west-2.amazonaws.com/Releases/SampleFaces.mp4) and [here](https://vector-blox-model-zoo.s3.us-west-2.amazonaws.com/Releases/SamplePlates.mp4).

