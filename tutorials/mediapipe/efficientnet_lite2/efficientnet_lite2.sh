
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

echo "Downloading efficientnet_lite2..."
# model details @ https://developers.google.com/mediapipe/solutions/vision/image_classifier#efficientnet-lite0_model_recommended
if [ ! -f efficientnet_lite2.tflite ]; then
    wget -q --no-check-certificate https://storage.googleapis.com/mediapipe-models/image_classifier/efficientnet_lite2/int8/latest/efficientnet_lite2.tflite
fi

if [ -f efficientnet_lite2.tflite ]; then
   tflite_preprocess efficientnet_lite2.tflite  --scale 255
fi

if [ -f efficientnet_lite2.pre.tflite ]; then
    echo "Generating VNNX for V1000 configuration..."
    vnnx_compile -c V1000 -t efficientnet_lite2.pre.tflite -o efficientnet_lite2.vnnx
fi

if [ -f efficientnet_lite2.vnnx ]; then
    echo "Running Simulation..."
    python $VBX_SDK/example/python/classifier.py efficientnet_lite2.vnnx $VBX_SDK/tutorials/test_images/oreo.jpg 
    echo "C Simulation Command:"
    echo '$VBX_SDK/example/sim-c/sim-run-model efficientnet_lite2.vnnx $VBX_SDK/tutorials/test_images/oreo.jpg CLASSIFY'
fi

deactivate
