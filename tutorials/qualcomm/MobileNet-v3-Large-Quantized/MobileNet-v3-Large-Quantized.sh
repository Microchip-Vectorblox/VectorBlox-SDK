
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

echo "Downloading MobileNet-v3-Large-Quantized..."
# model details @ https://huggingface.co/qualcomm/MobileNet-v3-Large-Quantized
if [ ! -f MobileNet-v3-Large-Quantized.tflite ]; then
    wget -q --no-check-certificate https://huggingface.co/qualcomm/MobileNet-v3-Large-Quantized/resolve/main/MobileNet-v3-Large-Quantized.tflite
fi

if [ -f MobileNet-v3-Large-Quantized.tflite ]; then
   tflite_preprocess MobileNet-v3-Large-Quantized.tflite  --scale 255
fi

if [ -f MobileNet-v3-Large-Quantized.pre.tflite ]; then
    echo "Generating VNNX for V1000 configuration..."
    vnnx_compile -c V1000 -t MobileNet-v3-Large-Quantized.pre.tflite -o MobileNet-v3-Large-Quantized.vnnx
fi

if [ -f MobileNet-v3-Large-Quantized.vnnx ]; then
    echo "Running Simulation..."
    python $VBX_SDK/example/python/classifier.py MobileNet-v3-Large-Quantized.vnnx $VBX_SDK/tutorials/test_images/oreo.jpg 
    echo "C Simulation Command:"
    echo '$VBX_SDK/example/sim-c/sim-run-model MobileNet-v3-Large-Quantized.vnnx $VBX_SDK/tutorials/test_images/oreo.jpg CLASSIFY'
fi

deactivate
