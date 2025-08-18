
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

echo "Checking for Numpy calibration data file..."
if [ ! -f $VBX_SDK/tutorials/imagenetv2_rgb_norm_20x224x224x3.npy ]; then
    generate_npy $VBX_SDK/tutorials/imagenetv2_rgb_20x224x224x3.npy -o $VBX_SDK/tutorials/imagenetv2_rgb_norm_20x224x224x3.npy -s 300 300  --norm 
fi

echo "Checking for efficientnet-lite4-int8 files..."

# model details @ https://www.kaggle.com/models/google/efficientnet/tensorFlow1/b4-classification/1
if [ ! -f efficientnet-lite4-int8.tflite ]; then
    wget -q --no-check-certificate https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/lite/efficientnet-lite4.tar.gz
   tar -xzf efficientnet-lite4.tar.gz
   cp efficientnet-lite4/efficientnet-lite4-int8.tflite .
fi


if [ -f efficientnet-lite4-int8.tflite ]; then 
   echo "Cutting graph" 
   tflite_cut efficientnet-lite4-int8.tflite -c 0 118
   mv efficientnet-lite4-int8.1.tflite efficientnet-lite4-int8.cut.tflite 
fi

if [ -f efficientnet-lite4-int8.cut.tflite ]; then
   tflite_preprocess efficientnet-lite4-int8.cut.tflite  --scale 255
fi

if [ -f efficientnet-lite4-int8.cut.pre.tflite ]; then
    echo "Generating VNNX for V1000 configuration..."
    vnnx_compile -c V1000 -t efficientnet-lite4-int8.cut.pre.tflite -o efficientnet-lite4-int8.vnnx
fi

if [ -f efficientnet-lite4-int8.vnnx ]; then
    echo "Running Simulation..."
    python $VBX_SDK/example/python/classifier.py efficientnet-lite4-int8.vnnx $VBX_SDK/tutorials/test_images/oreo.jpg 
    echo "C Simulation Command:"
    echo '$VBX_SDK/example/sim-c/sim-run-model efficientnet-lite4-int8.vnnx $VBX_SDK/tutorials/test_images/oreo.jpg CLASSIFY'
fi

deactivate
