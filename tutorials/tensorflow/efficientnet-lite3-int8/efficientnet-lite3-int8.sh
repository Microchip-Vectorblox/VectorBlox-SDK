
##########################################################
#  _    __          __             ____  __              #
# | |  / /__  _____/ /_____  _____/ __ )/ /___  _  __    #
# | | / / _ \/ ___/ __/ __ \/ ___/ __  / / __ \| |/_/    #
# | |/ /  __/ /__/ /_/ /_/ / /  / /_/ / / /_/ />  <      #
# |___/\___/\___/\__/\____/_/  /_____/_/\____/_/|_|      #
#                                                        #
# https://github.com/Microchip-Vectorblox/VectorBlox-SDK #
# v2.0                                                   #
#                                                        #
##########################################################

set -e
echo "Checking and activating VBX Python Environment..."
if [ -z $VBX_SDK ]; then
    echo "\$VBX_SDK not set. Please run 'source setup_vars.sh' from the SDK's root folder" && exit 1
fi
source $VBX_SDK/vbx_env/bin/activate

echo "Checking for Numpy calibration data file..."
if [ ! -f $VBX_SDK/tutorials/imagenetv2_rgb_norm_20x224x224x3.npy ]; then
    wget -P $VBX_SDK/tutorials/ https://vector-blox-model-zoo.s3.us-west-2.amazonaws.com/EAP/calib_npy/imagenetv2_rgb_norm_20x224x224x3.npy
fi

echo "Downloading efficientnet-lite3-int8..."
# model details @ https://www.kaggle.com/models/google/efficientnet/tensorFlow1/b3-classification/1
if [ ! -f efficientnet-lite3.tar.gz ]; then
    wget -q --no-check-certificate https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/lite/efficientnet-lite3.tar.gz
fi
tar -xzf efficientnet-lite3.tar.gz
cp efficientnet-lite3/efficientnet-lite3-int8.tflite .
tflite_cut efficientnet-lite3-int8.tflite -c 0
mv efficientnet-lite3-int8.1.tflite efficientnet-lite3-int8.tflite
tflite_cut efficientnet-lite3-int8.tflite -c 93
mv efficientnet-lite3-int8.0.tflite efficientnet-lite3-int8.tflite

if [ -f efficientnet-lite3-int8.tflite ]; then
   tflite_preprocess efficientnet-lite3-int8.tflite  --scale 255
fi

if [ -f efficientnet-lite3-int8.pre.tflite ]; then
    echo "Generating VNNX for V1000 configuration..."
    vnnx_compile -c V1000 -t efficientnet-lite3-int8.pre.tflite -o efficientnet-lite3-int8.vnnx
fi

if [ -f efficientnet-lite3-int8.vnnx ]; then
    echo "Running Simulation..."
    python $VBX_SDK/example/python/classifier.py efficientnet-lite3-int8.vnnx $VBX_SDK/tutorials/test_images/oreo.jpg 
    echo "C Simulation Command:"
    echo '$VBX_SDK/example/sim-c/sim-run-model efficientnet-lite3-int8.vnnx $VBX_SDK/tutorials/test_images/oreo.jpg CLASSIFY'
fi

deactivate
