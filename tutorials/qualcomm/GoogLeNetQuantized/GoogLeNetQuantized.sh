
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

echo "Downloading GoogLeNetQuantized..."
# model details @ https://aihub.qualcomm.com/models/googlenetquantized
if [ ! -f GoogLeNetQuantized.tflite ]; then
    wget -q --no-check-certificate https://huggingface.co/qualcomm/GoogLeNetQuantized/resolve/main/GoogLeNetQuantized.tflite
fi

if [ -f GoogLeNetQuantized.tflite ]; then
   tflite_preprocess GoogLeNetQuantized.tflite  --scale 255
fi

if [ -f GoogLeNetQuantized.pre.tflite ]; then
    echo "Generating VNNX for V1000 configuration..."
    vnnx_compile -c V1000 -t GoogLeNetQuantized.pre.tflite -o GoogLeNetQuantized.vnnx
fi

if [ -f GoogLeNetQuantized.vnnx ]; then
    echo "Running Simulation..."
    python $VBX_SDK/example/python/classifier.py GoogLeNetQuantized.vnnx $VBX_SDK/tutorials/test_images/oreo.jpg 
    echo "C Simulation Command:"
    echo '$VBX_SDK/example/sim-c/sim-run-model GoogLeNetQuantized.vnnx $VBX_SDK/tutorials/test_images/oreo.jpg CLASSIFY'
fi

deactivate
