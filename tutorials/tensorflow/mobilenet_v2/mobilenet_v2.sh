
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
if [ ! -f $VBX_SDK/tutorials/imagenetv2_rgb_20x224x224x3.npy ]; then
    generate_npy $VBX_SDK/tutorials/imagenetv2_rgb_20x224x224x3.npy -o $VBX_SDK/tutorials/imagenetv2_rgb_20x224x224x3.npy -s 224 224 
fi

echo "Downloading mobilenet_v2..."
# model details @ https://keras.io/api/applications/mobilenet/
python - <<EOF
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
mobilenet_v2_model = MobileNetV2(input_shape=(224, 224, 3), weights='imagenet', classifier_activation=None)
mobilenet_v2_model.save('saved_model/')
EOF

echo "Generating TF Lite..."
tflite_quantize saved_model mobilenet_v2.tflite -d $VBX_SDK/tutorials/imagenetv2_rgb_20x224x224x3.npy \
--mean 127.5 \
--scale 127.5 --shape 1 224 224 3

if [ -f mobilenet_v2.tflite ]; then
   tflite_preprocess mobilenet_v2.tflite  --mean 127.5 --scale 127.5
fi

if [ -f mobilenet_v2.pre.tflite ]; then
    echo "Generating VNNX for V1000 configuration..."
    vnnx_compile -c V1000 -t mobilenet_v2.pre.tflite -o mobilenet_v2.vnnx
fi

if [ -f mobilenet_v2.vnnx ]; then
    echo "Running Simulation..."
    python $VBX_SDK/example/python/classifier.py mobilenet_v2.vnnx $VBX_SDK/tutorials/test_images/oreo.jpg 
    echo "C Simulation Command:"
    echo '$VBX_SDK/example/sim-c/sim-run-model mobilenet_v2.vnnx $VBX_SDK/tutorials/test_images/oreo.jpg CLASSIFY'
fi

deactivate
