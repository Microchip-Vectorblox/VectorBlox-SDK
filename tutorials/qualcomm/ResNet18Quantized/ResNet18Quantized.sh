
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

echo "Downloading ResNet18Quantized..."
# model details @ https://aihub.qualcomm.com/models/resnet18quantized
if [ ! -f ResNet18Quantized.tflite ]; then
    wget -q --no-check-certificate https://huggingface.co/qualcomm/ResNet18Quantized/resolve/main/ResNet18Quantized.tflite
fi

if [ -f ResNet18Quantized.tflite ]; then
   tflite_preprocess ResNet18Quantized.tflite  --scale 255
fi

if [ -f ResNet18Quantized.pre.tflite ]; then
    echo "Generating VNNX for V1000 configuration..."
    vnnx_compile -c V1000 -t ResNet18Quantized.pre.tflite -o ResNet18Quantized.vnnx
fi

if [ -f ResNet18Quantized.vnnx ]; then
    echo "Running Simulation..."
    python $VBX_SDK/example/python/classifier.py ResNet18Quantized.vnnx $VBX_SDK/tutorials/test_images/oreo.jpg 
    echo "C Simulation Command:"
    echo '$VBX_SDK/example/sim-c/sim-run-model ResNet18Quantized.vnnx $VBX_SDK/tutorials/test_images/oreo.jpg CLASSIFY'
fi

deactivate
