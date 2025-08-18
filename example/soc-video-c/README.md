## Pre-requisites

- PolarFire SoC Video Kit, running latest [VectorBlox Demo design](https://github.com/Microchip-Vectorblox/VectorBlox-SoC-Video-Kit-Demo/releases), which uses [Yocto 2023.02.1](https://github.com/polarfire-soc/meta-polarfire-soc-yocto-bsp/releases/download/v2023.02.1/core-image-minimal-dev-mpfs-video-kit-20230328105837.rootfs.wic.gz)
- Logged into Yocto as `root` with ethernet connection
- In the `/home/root` directory, download the quick start script:
    ```
    wget --no-check-certificate https://raw.githubusercontent.com/Microchip-Vectorblox/assets/refs/heads/main/quick_start.sh
    ```
- Run the script with the realease version you want to download. The command below shows how to download the `2.0.2` release (if no version is specified it will download the latest):
    ```
    bash quick_start.sh 2.0.2
    ```
    The `quick_start.sh` script will download the `VectorBlox-SDK-release-v2.0.2`, the `samples_V1000_2.0.2` sample networks, the `camera_setup.zip` camera setup and unzip them.
    
    Before continuing, make sure that all the files have been downloaded properly and unzipped succesfully.


## Starting the VectorBlox demo on the PolarFire SoC Video Kit
- Move to the `soc-video-c` directory:
    ```
    cd VectorBlox-SDK-release-v2.0.2/example/soc-video-c
    ```
- Run `make overlay` to add the VectorBlox instance to the device tree (required every boot), unless the setup_
- Run `make` to build the demo application
- Run `./run-video-model` to launch the demo

## Controlling the VectorBlox HDMI demo on the PolarFire SoC Video Kit

- The demo consists of the following:
    - Face Recognition Demo
    - Classification (Mobilenetv2)
    - Object Detection (Yolov8n) 
    - Pose Estimation (Yolov8n Pose)
    - Semantic Segmentation (FFNet-122NS)
    - Depth Estimation (Midas V2)
    
- Use the `ENTER` key to switch modes. Entering `q` (pressing `q` and `ENTER`) quits the demo
- In the `Face Recognition` mode, you can enter `a` to add or `d` to delete face embeddings
    - Entering `a` initially highlights the largest face on-screen, entering `a` again adds that face to the embeddings. You will then be prompted to enter a name( or just press `ENTER` to use the default ID)

    - Entering `d` will list the indices and names of the embeddings. Enter the desired index to delete the specified embedding from the database (or press `ENTER` to skip the deletion)

- Entering `b` on any models that use Pose Estimation for postprocessing will allow the user to toggle between blackout options for the img output.


Samples videos for input to the Faces Recognition modes are available [here](https://github.com/Microchip-Vectorblox/assets/releases/download/assets/SampleFaces.mp4).

