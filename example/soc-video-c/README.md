## Pre-requisites

- Follow steps on [VectorBlox SoC Demo Design](https://github.com/Microchip-Vectorblox/VectorBlox-SoC-Video-Kit-Demo), including the step to Build the demo.

## Starting the VectorBlox demo on the PolarFire SoC Video Kit
- Move to the `soc-video-c` directory based on release version number:
    ```
    cd VectorBlox-SDK-release-v<VERSION#>/example/soc-video-c
    ```
- Run `make overlay` to add the VectorBlox instance to the device tree (required every boot), unless the setup_
- Run `make` to build the demo application
- Run `./run-video-model` to launch the demo

## Controlling the VectorBlox Demo on the PolarFire SoC Video Kit
To interact with the VectorBlox Video demo the following can be done:
    
- Use the `ENTER` key to switch models. Entering `q` (pressing `q` and `ENTER`) quits the demo.
- In the `Recognition` mode, you can enter `a` to add or `d` to delete face embeddings.
    - Entering `a` initially highlights the largest face on-screen, and entering `a` again adds that face to the embeddings. You will then be prompted to enter a name( or just press `ENTER` to use the default ID)

    - Entering `d` will list the indices and names of the embeddings. Enter the desired index to delete the specified embedding from the database (or press `ENTER` to skip the deletion)

- Entering `b` on any models that use Pose Estimation for postprocessing will allow the user to toggle between blackout options for the img output.


Sample videos for input to the Face Recognition mode are available [here](https://github.com/Microchip-Vectorblox/assets/releases/download/assets/SampleFaces.mp4).

