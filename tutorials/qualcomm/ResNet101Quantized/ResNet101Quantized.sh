
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

echo "Checking for ResNet101Quantized files..."

# model details @ https://huggingface.co/qualcomm/ResNet101
if [ ! -f ResNet101Quantized.tflite ]; then
    wget -q --no-check-certificate https://huggingface.co/qualcomm/ResNet101/resolve/v0.32.0/ResNet101_w8a8.tflite
    mv ResNet101_w8a8.tflite ResNet101Quantized.tflite
fi


if [ -f ResNet101Quantized.tflite ]; then
   tflite_preprocess ResNet101Quantized.tflite  --scale 255
fi

if [ -f ResNet101Quantized.pre.tflite ]; then
    echo "Generating VNNX for V1000 configuration..."
    vnnx_compile -c V1000 -t ResNet101Quantized.pre.tflite -o ResNet101Quantized.vnnx
fi

if [ -f ResNet101Quantized.vnnx ]; then
    echo "Running Simulation..."
    python $VBX_SDK/example/python/classifier.py ResNet101Quantized.vnnx $VBX_SDK/tutorials/test_images/oreo.jpg 
    echo "C Simulation Command:"
    echo '$VBX_SDK/example/sim-c/sim-run-model ResNet101Quantized.vnnx $VBX_SDK/tutorials/test_images/oreo.jpg CLASSIFY'
fi

deactivate
