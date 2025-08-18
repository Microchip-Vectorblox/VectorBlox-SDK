
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

echo "Checking for efficientnet-lite0 files..."

# model details @ https://www.kaggle.com/models/tensorflow/efficientdet/tfLite
if [ ! -f efficientnet-lite0.tflite ]; then
    curl -L -o model.tar.gz https://www.kaggle.com/api/v1/models/tensorflow/efficientnet/tfLite/lite0-int8/2/download
    tar -xzf model.tar.gz
    cp 2.tflite efficientnet-lite0.tflite
fi


if [ -f efficientnet-lite0.tflite ]; then
   tflite_preprocess efficientnet-lite0.tflite  --scale 255
fi

if [ -f efficientnet-lite0.pre.tflite ]; then
    echo "Generating VNNX for V1000 configuration..."
    vnnx_compile -c V1000 -t efficientnet-lite0.pre.tflite -o efficientnet-lite0.vnnx
fi

if [ -f efficientnet-lite0.vnnx ]; then
    echo "Running Simulation..."
    python $VBX_SDK/example/python/classifier.py efficientnet-lite0.vnnx $VBX_SDK/tutorials/test_images/oreo.jpg 
    echo "C Simulation Command:"
    echo '$VBX_SDK/example/sim-c/sim-run-model efficientnet-lite0.vnnx $VBX_SDK/tutorials/test_images/oreo.jpg CLASSIFY'
fi

deactivate
