
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
    wget -P $VBX_SDK/tutorials/ https://vector-blox-model-zoo.s3.us-west-2.amazonaws.com/EAP/calib_npy/imagenetv2_rgb_norm_20x224x224x3.npy
fi

echo "Downloading torchvision_resnet50..."
# model details @ https://pytorch.org/vision/0.14/models/generated/torchvision.models.resnet50.html#torchvision.models.resnet50
python $VBX_SDK/tutorials/torchvision_to_onnx.py resnet50
if [ ! -f calibration_image_sample_data_20x128x128x3_float32.npy ]; then
    wget https://vector-blox-model-zoo.s3.us-west-2.amazonaws.com/EAP/calib_npy/calibration_image_sample_data_20x128x128x3_float32.npy
fi

echo "Running ONNX2TF..."
onnx2tf -cind input.1 $VBX_SDK/tutorials/imagenetv2_rgb_norm_20x224x224x3.npy [[[[0.485,0.456,0.406]]]] [[[[0.229,0.224,0.225]]]] \
-i resnet50.onnx \
--output_signaturedefs \
--output_integer_quantized_tflite
cp saved_model/resnet50_full_integer_quant.tflite torchvision_resnet50.tflite

if [ -f torchvision_resnet50.tflite ]; then
   tflite_preprocess torchvision_resnet50.tflite  --mean 123.675 116.28 103.53 --scale 58.4 57.1 57.38
fi

if [ -f torchvision_resnet50.pre.tflite ]; then
    echo "Generating VNNX for V1000 configuration..."
    vnnx_compile -c V1000 -t torchvision_resnet50.pre.tflite -o torchvision_resnet50.vnnx
fi

if [ -f torchvision_resnet50.vnnx ]; then
    echo "Running Simulation..."
    python $VBX_SDK/example/python/classifier.py torchvision_resnet50.vnnx $VBX_SDK/tutorials/test_images/oreo.jpg 
    echo "C Simulation Command:"
    echo '$VBX_SDK/example/sim-c/sim-run-model torchvision_resnet50.vnnx $VBX_SDK/tutorials/test_images/oreo.jpg CLASSIFY'
fi

deactivate
