
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
if [ ! -f $VBX_SDK/tutorials/imagenetv2_rgb_norm_20x227x227x3.npy ]; then
    generate_npy $VBX_SDK/tutorials/imagenetv2_rgb_20x224x224x3.npy -o $VBX_SDK/tutorials/imagenetv2_rgb_norm_20x227x227x3.npy -s 227 227  --norm 
fi

echo "Checking for torchvision_squeezenet1_0 files..."
if [ ! -f torchvision_squeezenet1_0.tflite ]; then
# model details @ https://pytorch.org/vision/0.14/models/generated/torchvision.models.squeezenet1_0.html#torchvision.models.squeezenet1_0
python $VBX_SDK/tutorials/torchvision_to_onnx.py squeezenet1_0 -i 227
fi

if [ ! -f torchvision_squeezenet1_0.tflite ]; then
   echo "Running ONNX2TF..."
   onnx2tf -cind input.1 $VBX_SDK/tutorials/imagenetv2_rgb_norm_20x227x227x3.npy [[[[0.485,0.456,0.406]]]] [[[[0.229,0.224,0.225]]]] \
-i squeezenet1_0.onnx \
--output_signaturedefs \
--output_integer_quantized_tflite
   cp saved_model/squeezenet1_0_full_integer_quant.tflite torchvision_squeezenet1_0.tflite
fi
if [ -f torchvision_squeezenet1_0.tflite ]; then
   tflite_preprocess torchvision_squeezenet1_0.tflite  --mean 123.675 116.28 103.53 --scale 58.4 57.1 57.38
fi

if [ -f torchvision_squeezenet1_0.pre.tflite ]; then
    echo "Generating VNNX for V1000 configuration..."
    vnnx_compile -c V1000 -t torchvision_squeezenet1_0.pre.tflite -o torchvision_squeezenet1_0.vnnx
fi

if [ -f torchvision_squeezenet1_0.vnnx ]; then
    echo "Running Simulation..."
    python $VBX_SDK/example/python/classifier.py torchvision_squeezenet1_0.vnnx $VBX_SDK/tutorials/test_images/oreo.jpg 
    echo "C Simulation Command:"
    echo '$VBX_SDK/example/sim-c/sim-run-model torchvision_squeezenet1_0.vnnx $VBX_SDK/tutorials/test_images/oreo.jpg CLASSIFY'
fi

deactivate
