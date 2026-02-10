
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

echo "Checking for resnet18_86s_07p files..."

# model details @ 
if [ ! -f resnet18_86s_07p.tflite ]; then
    wget -q --no-check-certificate https://github.com/Microchip-Vectorblox/assets/releases/download/assets/resnet18_86s_07p.tflite
fi


if [ -f resnet18_86s_07p.tflite ]; then
    echo "Generating VNNX for V1000 ucomp configuration..."
    vnnx_compile -s V1000 -c ucomp -t resnet18_86s_07p.tflite --uint8 --mean 123.675 116.28 103.53 --scale 58.4 57.1 57.38  -o resnet18_86s_07p.ucomp
fi

if [ -f resnet18_86s_07p.ucomp ]; then
    echo "Running Simulation..."
    python $VBX_SDK/example/python/classifier.py resnet18_86s_07p.ucomp $VBX_SDK/tutorials/test_images/oreo.jpg 
    echo "C Simulation Command:"
    echo '$VBX_SDK/example/sim-c/sim-run-model resnet18_86s_07p.ucomp $VBX_SDK/tutorials/test_images/oreo.jpg CLASSIFY'
fi

deactivate
