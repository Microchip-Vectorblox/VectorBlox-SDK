# PolarFire SoC Video Kit VectorBlox Video Pipeline Demo

## Pre-requisites

- Follow steps on [VectorBlox SoC Demo Design](https://github.com/Microchip-Vectorblox/VectorBlox-SoC-Video-Kit-Demo), including the step to build the demo.

## Starting the VectorBlox demo on the PolarFire SoC Video Kit

Getting Started with the VectorBlox Demo on the PolarFire SoC Video Kit  
Once you have logged in, navigate to the soc-video-c directory that matches your release version number.  

- Run `make overlay` to add the VectorBlox instance to the device tree (required every boot). If make overlay reports error `Overlay mpfs_vbx.dtbo already exists`, this can be ignored.
- Run `make` to build the demo application.
- Run `./run-video-model` to launch the demo.

## Controlling the VectorBlox demo on the PolarFire SoC Video Kit

To interact with the VectorBlox video demo:

- Use the `ENTER` key to switch models. Entering `q` (pressing `q` then `ENTER`) quits the demo.
- In `Recognition` mode, enter `a` to add or `d` to delete face embeddings.
  - Entering `a` initially highlights the largest face on-screen. Entering `a` again adds that face to the embeddings. You will then be prompted to enter a name (or press `ENTER` to use the default ID).
  - Entering `d` lists the indices and names of the embeddings. Enter the desired index to delete the specified embedding from the database (or press `ENTER` to skip deletion).
- Entering `b` on any model that uses Pose Estimation for post-processing toggles the blackout options for the image output.

Sample videos for input to the Face Recognition mode are available [here](https://github.com/Microchip-Vectorblox/assets/releases/download/assets/SampleFaces.mp4).
