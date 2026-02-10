
##########################################################
#  _    __          __             ____  __              #
# | |  / /__  _____/ /_____  _____/ __ )/ /___  _  __    #
# | | / / _ \/ ___/ __/ __ \/ ___/ __  / / __ \| |/_/    #
# | |/ /  __/ /__/ /_/ /_/ / /  / /_/ / / /_/ />  <      #
# |___/\___/\___/\__/\____/_/  /_____/_/\____/_/|_|      #
#                                                        #
# https://github.com/Microchip-Vectorblox/VectorBlox-SDK #
# v3.0                                                   #
#                                                        #
##########################################################

set -e
echo "Checking and activating VBX Python Environment..."
if [ -z $VBX_SDK ]; then
    echo "\$VBX_SDK not set. Please run 'source setup_vars.sh' from the SDK's root folder" && exit 1
fi
source $VBX_SDK/vbx_env/bin/activate

echo "Checking for QuickSRNetMedium-Quantized files..."

# model details @ https://aihub.qualcomm.com/models/quicksrnetmedium_quantized
if [ ! -f QuickSRNetMedium-Quantized.tflite ]; then
    wget -q --no-check-certificate https://huggingface.co/qualcomm/QuickSRNetMedium/resolve/v0.29.1/QuickSRNetMedium_w8a8.tflite
    mv QuickSRNetMedium_w8a8.tflite QuickSRNetMedium-Quantized.tflite
fi


if [ -f QuickSRNetMedium-Quantized.tflite ]; then
   tflite_preprocess QuickSRNetMedium-Quantized.tflite  --scale 255
fi

if [ -f QuickSRNetMedium-Quantized.pre.tflite ]; then
    echo "Generating VNNX for V1000 ncomp configuration..."
    vnnx_compile -s V1000 -c ncomp -t QuickSRNetMedium-Quantized.pre.tflite  -o QuickSRNetMedium-Quantized_V1000_ncomp.vnnx
fi

if [ -f QuickSRNetMedium-Quantized_V1000_ncomp.vnnx ]; then
    echo "Running Simulation..."
    python $VBX_SDK/example/python/super_resolution.py QuickSRNetMedium-Quantized_V1000_ncomp.vnnx $VBX_SDK/tutorials/test_images/oreo.jpg 
    echo "C Simulation Command:"
    echo '$VBX_SDK/example/sim-c/sim-run-model QuickSRNetMedium-Quantized_V1000_ncomp.vnnx $VBX_SDK/tutorials/test_images/oreo.jpg  '
fi

deactivate
