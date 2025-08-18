
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

echo "Checking for WideResNet50-Quantized files..."

# model details @ https://huggingface.co/qualcomm/WideResNet50
if [ ! -f WideResNet50-Quantized.tflite ]; then
    wget -q --no-check-certificate https://huggingface.co/qualcomm/WideResNet50/blob/v0.32.0/WideResNet50.onnx
fi


if [ ! -f WideResNet50-Quantized.tflite ]; then
   echo "Running ONNX2TF..."
   onnx2tf -cind image_tensor $VBX_SDK/tutorials/coco2017_rgb_norm_20x224x224x3.npy [[[0.,0.,0.]]] [[[1.,1.,1.]]] \
-i WideResNet50.onnx \
--output_signaturedefs \
--output_integer_quantized_tflite
   cp saved_model/WideResNet50_full_integer_quant.tflite WideResNet50-Quantized.tflite
fi
if [ -f WideResNet50-Quantized.tflite ]; then
   tflite_preprocess WideResNet50-Quantized.tflite  --scale 255
fi

if [ -f WideResNet50-Quantized.pre.tflite ]; then
    echo "Generating VNNX for V1000 configuration..."
    vnnx_compile -c V1000 -t WideResNet50-Quantized.pre.tflite -o WideResNet50-Quantized.vnnx
fi

if [ -f WideResNet50-Quantized.vnnx ]; then
    echo "Running Simulation..."
    python $VBX_SDK/example/python/classifier.py WideResNet50-Quantized.vnnx $VBX_SDK/tutorials/test_images/oreo.jpg 
    echo "C Simulation Command:"
    echo '$VBX_SDK/example/sim-c/sim-run-model WideResNet50-Quantized.vnnx $VBX_SDK/tutorials/test_images/oreo.jpg CLASSIFY'
fi

deactivate
