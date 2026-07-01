
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
if [ ! -f $VBX_SDK/tutorials/imagenetv2_rgb_20x224x224x3.npy ]; then
    generate_npy $VBX_SDK/tutorials/imagenetv2_rgb_20x224x224x3.npy -o $VBX_SDK/tutorials/imagenetv2_rgb_20x224x224x3.npy -s 224 224 
fi

echo "Checking for mobilenet_v2 files..."

# model details @ https://keras.io/api/applications/mobilenet/
if [ ! -f mobilenet_v2.tflite ]; then
python - <<EOF
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
mobilenet_v2_model = MobileNetV2(input_shape=(224, 224, 3), weights='imagenet', classifier_activation=None)
mobilenet_v2_model.save('saved_model/')
EOF
fi



if [ -f saved_model/saved_model.pb ]; then
   if ! echo "aa755feab52b9d4302779caf4ee70554 saved_model/saved_model.pb" | md5sum -c; then
       echo -e "\n There is an issue with the mobilenet_v2 model file as the expected checksum does not match.\n The model source can be found at: https://keras.io/api/applications/mobilenet/.\n If the model information has changed, please update this script and re-run the tutorial."
       exit 1
   fi
fi


# tflite_quantize is an internal tool designed to quantize a saved_model directory using the npy array
#  Purpose: Convert source model directory to int8 tflite format
#  - Required Inputs: model source directory, calibration npy array
#  - Output: int8 tflite model
if [ ! -f mobilenet_v2.tflite ]; then
   echo "Generating TF Lite..."
   tflite_quantize saved_model mobilenet_v2.tflite -d $VBX_SDK/tutorials/imagenetv2_rgb_20x224x224x3.npy \
--mean 127.5 \
--scale 127.5 --shape 1 224 224 3
fi


# tflite_preprocess is an internal tool used to add a preprocess layer to the start of the model
#  Purpose: adds a preprocess layer to the start of the model (if none, will just preprocess by adding a uint8->int8 layer)
#  - Required Inputs: tflite source model, additional arguments 
#  - Outputs: preprocessed tflite model
if [ -f mobilenet_v2.tflite ]; then
   tflite_preprocess mobilenet_v2.tflite  --mean 127.5 --scale 127.5
fi


# vnnx_compile is an internal tool that converts an int8 tflite file to a binary file that can be run on the SDK and VectorBlox FPGA
#  Purpose: converts int8 tflite to binary
#  - Required Inputs: int8 tflite, size configuration, compression configuration, output file name
#  - Outputs: binary object files(.hex and binary file)
if [ -f mobilenet_v2.pre.tflite ]; then
    echo "Generating VNNX for V1000 ncomp configuration..."
    vnnx_compile -s V1000 -c ncomp -t mobilenet_v2.pre.tflite  -o mobilenet_v2_V1000_ncomp.vnnx
fi


# This step runs the final compiled binary in Python, it also shows how to run the same file in C simulation for SDK
#   *Currently C simulation is not supported for unstructured compression
if [ -f mobilenet_v2_V1000_ncomp.vnnx ]; then
    echo "Running Simulation..."
    python $VBX_SDK/example/python/classifier.py mobilenet_v2_V1000_ncomp.vnnx $VBX_SDK/tutorials/test_images/oreo.jpg 
    echo "C Simulation Command:"
    echo '$VBX_SDK/example/sim-c/sim-run-model mobilenet_v2_V1000_ncomp.vnnx $VBX_SDK/tutorials/test_images/oreo.jpg CLASSIFY'
fi

deactivate
