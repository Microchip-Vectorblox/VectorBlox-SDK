
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

echo "Checking for Numpy calibration data file..."
if [ ! -f $VBX_SDK/tutorials/imagenetv2_rgb_norm_20x224x224x3.npy ]; then
    generate_npy $VBX_SDK/tutorials/imagenetv2_rgb_20x224x224x3.npy -o $VBX_SDK/tutorials/imagenetv2_rgb_norm_20x224x224x3.npy -s 224 224  --norm 
fi

echo "Checking for onnx_squeezenet1.1 files..."
if [ ! -f onnx_squeezenet1.1.tflite ]; then
# model details @ https://github.com/onnx/models/tree/main/validated/vision/classification/squeezenet
wget -q --no-check-certificate https://media.githubusercontent.com/media/onnx/models/main/validated/vision/classification/squeezenet/model/squeezenet1.1-7.onnx
fi

if [ ! -f onnx_squeezenet1.1.tflite ]; then
   echo "Running ONNX2TF..."
   onnx2tf -cind data $VBX_SDK/tutorials/imagenetv2_rgb_norm_20x224x224x3.npy [[[[0.485,0.456,0.406]]]] [[[[0.229,0.224,0.225]]]] \
-i squeezenet1.1-7.onnx \
--output_signaturedefs \
--output_integer_quantized_tflite
   cp saved_model/squeezenet1.1-7_full_integer_quant.tflite onnx_squeezenet1.1.tflite
fi
if [ -f onnx_squeezenet1.1.tflite ]; then
   tflite_preprocess onnx_squeezenet1.1.tflite  --mean 123.675 116.28 103.53 --scale 58.4 57.1 57.38
fi

if [ -f onnx_squeezenet1.1.pre.tflite ]; then
    echo "Generating VNNX for V1000 ncomp configuration..."
    vnnx_compile -s V1000 -c ncomp -t onnx_squeezenet1.1.pre.tflite  -o onnx_squeezenet1.1_V1000_ncomp.vnnx
fi

if [ -f onnx_squeezenet1.1_V1000_ncomp.vnnx ]; then
    echo "Running Simulation..."
    python $VBX_SDK/example/python/classifier.py onnx_squeezenet1.1_V1000_ncomp.vnnx $VBX_SDK/tutorials/test_images/oreo.jpg 
    echo "C Simulation Command:"
    echo '$VBX_SDK/example/sim-c/sim-run-model onnx_squeezenet1.1_V1000_ncomp.vnnx $VBX_SDK/tutorials/test_images/oreo.jpg CLASSIFY'
fi

deactivate
