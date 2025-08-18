
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

echo "Checking for XLSR-Quantized files..."

# model details @ https://aihub.qualcomm.com/models/xlsr_quantized
if [ ! -f XLSR-Quantized.tflite ]; then
    wget -q --no-check-certificate https://huggingface.co/qualcomm/XLSR/resolve/v0.29.1/XLSR_w8a8.tflite
    mv XLSR_w8a8.tflite XLSR-Quantized.tflite
fi


if [ -f XLSR-Quantized.tflite ]; then
   tflite_preprocess XLSR-Quantized.tflite  --scale 255
fi

if [ -f XLSR-Quantized.pre.tflite ]; then
    echo "Generating VNNX for V1000 configuration..."
    vnnx_compile -c V1000 -t XLSR-Quantized.pre.tflite -o XLSR-Quantized.vnnx
fi

if [ -f XLSR-Quantized.vnnx ]; then
    echo "Running Simulation..."
    python $VBX_SDK/example/python/super_resolution.py XLSR-Quantized.vnnx $VBX_SDK/tutorials/test_images/oreo.jpg 
    echo "C Simulation Command:"
    echo '$VBX_SDK/example/sim-c/sim-run-model XLSR-Quantized.vnnx $VBX_SDK/tutorials/test_images/oreo.jpg  '
fi

deactivate
