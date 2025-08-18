
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

echo "Checking for QuickSRNetLarge-Quantized files..."

# model details @ https://aihub.qualcomm.com/models/quicksrnetlarge_quantized
if [ ! -f QuickSRNetLarge-Quantized.tflite ]; then
    wget -q --no-check-certificate https://huggingface.co/qualcomm/QuickSRNetLarge/resolve/v0.29.1/QuickSRNetLarge_w8a8.tflite
    mv QuickSRNetLarge_w8a8.tflite QuickSRNetLarge-Quantized.tflite
fi


if [ -f QuickSRNetLarge-Quantized.tflite ]; then
   tflite_preprocess QuickSRNetLarge-Quantized.tflite  --scale 255
fi

if [ -f QuickSRNetLarge-Quantized.pre.tflite ]; then
    echo "Generating VNNX for V1000 configuration..."
    vnnx_compile -c V1000 -t QuickSRNetLarge-Quantized.pre.tflite -o QuickSRNetLarge-Quantized.vnnx
fi

if [ -f QuickSRNetLarge-Quantized.vnnx ]; then
    echo "Running Simulation..."
    python $VBX_SDK/example/python/super_resolution.py QuickSRNetLarge-Quantized.vnnx $VBX_SDK/tutorials/test_images/oreo.jpg 
    echo "C Simulation Command:"
    echo '$VBX_SDK/example/sim-c/sim-run-model QuickSRNetLarge-Quantized.vnnx $VBX_SDK/tutorials/test_images/oreo.jpg  '
fi

deactivate
