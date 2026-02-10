
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

echo "Checking for 016_EfficientNet-lite files..."

# model details @ https://github.com/PINTO0309/PINTO_model_zoo
if [ ! -f 016_EfficientNet-lite.tflite ]; then
    mkdir -p temp
    cd temp
    wget -q https://raw.githubusercontent.com/PINTO0309/PINTO_model_zoo/refs/heads/main/016_EfficientNet-lite/download.sh
    bash download.sh
    cd ..
    pushd temp/016_EfficientNet-lite/01_efficientnet_lite0/03_integer_quantization; tar -zxf resources.tar.gz; popd
    cp temp/016_EfficientNet-lite/01_efficientnet_lite0/03_integer_quantization/efficientnet-lite0-int8.tflite 016_EfficientNet-lite.tflite
    rm -rf temp
fi


if [ -f 016_EfficientNet-lite.tflite ]; then
   tflite_preprocess 016_EfficientNet-lite.tflite  --scale 255
fi

if [ -f 016_EfficientNet-lite.pre.tflite ]; then
    echo "Generating VNNX for V1000 ncomp configuration..."
    vnnx_compile -s V1000 -c ncomp -t 016_EfficientNet-lite.pre.tflite  -o 016_EfficientNet-lite_V1000_ncomp.vnnx
fi

if [ -f 016_EfficientNet-lite_V1000_ncomp.vnnx ]; then
    echo "Running Simulation..."
    python $VBX_SDK/example/python/classifier.py 016_EfficientNet-lite_V1000_ncomp.vnnx $VBX_SDK/tutorials/test_images/oreo.jpg 
    echo "C Simulation Command:"
    echo '$VBX_SDK/example/sim-c/sim-run-model 016_EfficientNet-lite_V1000_ncomp.vnnx $VBX_SDK/tutorials/test_images/oreo.jpg CLASSIFY'
fi

deactivate
