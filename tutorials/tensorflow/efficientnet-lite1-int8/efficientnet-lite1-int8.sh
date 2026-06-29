
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


# generate_npy is an internal tool that creates a npy array
#  Purpose: Generates a npy array if an existing one does not exist, this is using custom img data
#  - Required Inputs: source dataset, output name, size
#  - Output: npy array
echo "Checking for Numpy calibration data file..."
if [ ! -f $VBX_SDK/tutorials/imagenetv2_rgb_norm_20x224x224x3.npy ]; then
    generate_npy $VBX_SDK/tutorials/imagenetv2_rgb_20x224x224x3.npy -o $VBX_SDK/tutorials/imagenetv2_rgb_norm_20x224x224x3.npy -s 240 240  --norm 
fi

echo "Checking for efficientnet-lite1-int8 files..."

# model details @ https://www.kaggle.com/models/google/efficientnet/tensorFlow1/b1-classification/1
if [ ! -f efficientnet-lite1-int8.tflite ]; then
    wget -q --no-check-certificate https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/lite/efficientnet-lite1.tar.gz
   tar -xzf efficientnet-lite1.tar.gz
   cp efficientnet-lite1/efficientnet-lite1-int8.tflite .
fi


if [ -f efficientnet-lite1.tar.gz ]; then
   if ! echo "6914aa625b58a2cb786f92597af8d63a efficientnet-lite1.tar.gz" | md5sum -c; then
       echo -e "\n There is an issue with the efficientnet-lite1-int8 model file as the expected checksum does not match.\n The model source can be found at: https://www.kaggle.com/models/google/efficientnet/tensorFlow1/b1-classification/1.\n If the model information has changed, please update this script and re-run the tutorial."
       exit 1
   fi
fi


# tflite_cut is an internal tool used to split an existing model into smaller models
#  Purpose: cuts a model into smaller subsections, can be used to decrease runtime or for debugging purposes
#  - Required Inputs: tflite source model, cut section(s)
#  - Outputs: tflite model
if [ -f efficientnet-lite1-int8.tflite ]; then 
   echo "Cutting graph" 
   tflite_cut efficientnet-lite1-int8.tflite -c 0 82
   mv efficientnet-lite1-int8.1.tflite efficientnet-lite1-int8.cut.tflite 
fi


# tflite_preprocess is an internal tool used to add a preprocess layer to the start of the model
#  Purpose: adds a preprocess layer to the start of the model (if none, will just preprocess by adding a uint8->int8 layer)
#  - Required Inputs: tflite source model, additional arguments 
#  - Outputs: preprocessed tflite model
if [ -f efficientnet-lite1-int8.cut.tflite ]; then
   tflite_preprocess efficientnet-lite1-int8.cut.tflite  --scale 255
fi


# vnnx_compile is an internal tool that converts an int8 tflite file to a binary file that can be run on the SDK and VectorBlox FPGA
#  Purpose: converts int8 tflite to binary
#  - Required Inputs: int8 tflite, size configuration, compression configuration, output file name
#  - Outputs: binary object files(.hex and binary file)
if [ -f efficientnet-lite1-int8.cut.pre.tflite ]; then
    echo "Generating VNNX for V1000 ncomp configuration..."
    vnnx_compile -s V1000 -c ncomp -t efficientnet-lite1-int8.cut.pre.tflite  -o efficientnet-lite1-int8_V1000_ncomp.vnnx
fi


# This step runs the final compiled binary in Python, it also shows how to run the same file in C simulation for SDK
#   *Currently C simulation is not supported for unstructured compression
if [ -f efficientnet-lite1-int8_V1000_ncomp.vnnx ]; then
    echo "Running Simulation..."
    python $VBX_SDK/example/python/classifier.py efficientnet-lite1-int8_V1000_ncomp.vnnx $VBX_SDK/tutorials/test_images/oreo.jpg 
    echo "C Simulation Command:"
    echo '$VBX_SDK/example/sim-c/sim-run-model efficientnet-lite1-int8_V1000_ncomp.vnnx $VBX_SDK/tutorials/test_images/oreo.jpg CLASSIFY'
fi

deactivate
