
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
    generate_npy $VBX_SDK/tutorials/imagenetv2_rgb_20x224x224x3.npy -o $VBX_SDK/tutorials/imagenetv2_rgb_norm_20x224x224x3.npy -s 224 224  --norm 
fi

echo "Downloading onnx_resnet18-v1..."
# model details @ https://github.com/onnx/models/tree/main/validated/vision/classification/resnet
wget -q --no-check-certificate https://media.githubusercontent.com/media/onnx/models/main/validated/vision/classification/resnet/model/resnet18-v1-7.onnx

echo "Running ONNX2TF..."
onnx2tf -cind data $VBX_SDK/tutorials/imagenetv2_rgb_norm_20x224x224x3.npy [[[[0.485,0.456,0.406]]]] [[[[0.229,0.224,0.225]]]] \
-i resnet18-v1-7.onnx \
--output_signaturedefs \
--output_integer_quantized_tflite
cp saved_model/resnet18-v1-7_full_integer_quant.tflite onnx_resnet18-v1.tflite

if [ -f onnx_resnet18-v1.tflite ]; then
   tflite_preprocess onnx_resnet18-v1.tflite  --mean 123.675 116.28 103.53 --scale 58.4 57.1 57.38
fi

if [ -f onnx_resnet18-v1.pre.tflite ]; then
    echo "Generating VNNX for V1000 configuration..."
    vnnx_compile -c V1000 -t onnx_resnet18-v1.pre.tflite -o onnx_resnet18-v1.vnnx
fi

if [ -f onnx_resnet18-v1.vnnx ]; then
    echo "Running Simulation..."
    python $VBX_SDK/example/python/classifier.py onnx_resnet18-v1.vnnx $VBX_SDK/tutorials/test_images/oreo.jpg 
    echo "C Simulation Command:"
    echo '$VBX_SDK/example/sim-c/sim-run-model onnx_resnet18-v1.vnnx $VBX_SDK/tutorials/test_images/oreo.jpg CLASSIFY'
fi

deactivate
