
##########################################################
#  _    __          __             ____  __              #
# | |  / /__  _____/ /_____  _____/ __ )/ /___  _  __    #
# | | / / _ \/ ___/ __/ __ \/ ___/ __  / / __ \| |/_/    #
# | |/ /  __/ /__/ /_/ /_/ / /  / /_/ / / /_/ />  <      #
# |___/\___/\___/\__/\____/_/  /_____/_/\____/_/|_|      #
#                                                        #
# https://github.com/Microchip-Vectorblox/VectorBlox-SDK #
# v3.1                                                   #
#                                                        #
##########################################################



set -e
echo "Checking and activating VBX Python Environment..."
if [ -z $VBX_SDK ]; then
    echo "\$VBX_SDK not set. Please run 'source setup_vars.sh' from the SDK's root folder" && exit 1
fi
source $VBX_SDK/vbx_env/bin/activate

echo "Checking for efficientnet-lite4 files..."

# model details @ https://www.kaggle.com/models/tensorflow/efficientdet/tfLite
if [ ! -f efficientnet-lite4.tflite ]; then
    curl -L -o model.tar.gz https://www.kaggle.com/api/v1/models/tensorflow/efficientnet/tfLite/lite4-int8/2/download
    tar -xzf model.tar.gz
    cp 2.tflite efficientnet-lite4.tflite
fi


if [ -f model.tar.gz ]; then
   if ! echo "fa705c71db83fde93708b6256bd89297 model.tar.gz" | md5sum -c; then
       echo -e "\n There is an issue with the efficientnet-lite4 model file as the expected checksum does not match.\n The model source can be found at: https://www.kaggle.com/models/tensorflow/efficientdet/tfLite.\n If the model information has changed, please update this script and re-run the tutorial."
       exit 1
   fi
fi


# tflite_preprocess is an internal tool used to add a preprocess layer to the start of the model
#  Purpose: adds a preprocess layer to the start of the model (if none, will just preprocess by adding a uint8->int8 layer)
#  - Required Inputs: tflite source model, additional arguments 
#  - Outputs: preprocessed tflite model
if [ -f efficientnet-lite4.tflite ]; then
   tflite_preprocess efficientnet-lite4.tflite  --scale 255
fi


# vnnx_compile is an internal tool that converts an int8 tflite file to a binary file that can be run on the SDK and VectorBlox FPGA
#  Purpose: converts int8 tflite to binary
#  - Required Inputs: int8 tflite, size configuration, compression configuration, output file name
#  - Outputs: binary object files(.hex and binary file)
if [ -f efficientnet-lite4.pre.tflite ]; then
    echo "Generating VNNX for V1000 ncomp configuration..."
    vnnx_compile -s V1000 -c ncomp -t efficientnet-lite4.pre.tflite  -o efficientnet-lite4_V1000_ncomp.vnnx
fi


# This step runs the final compiled binary in Python, it also shows how to run the same file in C simulation for SDK
#   *Currently C simulation is not supported for unstructured compression
if [ -f efficientnet-lite4_V1000_ncomp.vnnx ]; then
    echo "Running Simulation..."
    python $VBX_SDK/example/python/classifier.py efficientnet-lite4_V1000_ncomp.vnnx $VBX_SDK/tutorials/test_images/oreo.jpg 
    echo "C Simulation Command:"
    echo '$VBX_SDK/example/sim-c/sim-run-model efficientnet-lite4_V1000_ncomp.vnnx $VBX_SDK/tutorials/test_images/oreo.jpg CLASSIFY'
fi

deactivate
