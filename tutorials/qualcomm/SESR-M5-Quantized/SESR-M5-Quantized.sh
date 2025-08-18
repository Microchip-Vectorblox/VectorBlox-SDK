
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

echo "Checking for SESR-M5-Quantized files..."

# model details @ https://aihub.qualcomm.com/models/sesr_m5_quantized
if [ ! -f SESR-M5-Quantized.tflite ]; then
    wget -q --no-check-certificate https://huggingface.co/qualcomm/SESR-M5/resolve/v0.29.1/SESR-M5_w8a8.tflite
    mv SESR-M5_w8a8.tflite SESR-M5-Quantized.tflite
fi


if [ -f SESR-M5-Quantized.tflite ]; then
   tflite_preprocess SESR-M5-Quantized.tflite  --scale 255
fi

if [ -f SESR-M5-Quantized.pre.tflite ]; then
    echo "Generating VNNX for V1000 configuration..."
    vnnx_compile -c V1000 -t SESR-M5-Quantized.pre.tflite -o SESR-M5-Quantized.vnnx
fi

if [ -f SESR-M5-Quantized.vnnx ]; then
    echo "Running Simulation..."
    python $VBX_SDK/example/python/super_resolution.py SESR-M5-Quantized.vnnx $VBX_SDK/tutorials/test_images/oreo.jpg -i 
    echo "C Simulation Command:"
    echo '$VBX_SDK/example/sim-c/sim-run-model SESR-M5-Quantized.vnnx $VBX_SDK/tutorials/test_images/oreo.jpg  '
fi

deactivate
