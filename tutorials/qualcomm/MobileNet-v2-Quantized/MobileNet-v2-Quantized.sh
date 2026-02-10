
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
if [ ! -f $VBX_SDK/tutorials/coco2017_rgb_norm_20x224x224x3.npy ]; then
    generate_npy $VBX_SDK/tutorials/coco2017_rgb_20x416x416x3.npy -o $VBX_SDK/tutorials/coco2017_rgb_norm_20x224x224x3.npy -s 224 224  --norm 
fi

echo "Checking for MobileNet-v2-Quantized files..."

# model details @ https://aihub.qualcomm.com/models/mobilenet_v2
if [ ! -f MobileNet-v2-Quantized.tflite ]; then
    wget --no-check-certificate https://huggingface.co/qualcomm/MobileNet-v2/blob/v0.32.0/MobileNet-v2.onnx
fi


if [ ! -f MobileNet-v2-Quantized.tflite ]; then
   echo "Running ONNX2TF..."
   onnx2tf -cind image_tensor $VBX_SDK/tutorials/coco2017_rgb_norm_20x224x224x3.npy [[[0.,0.,0.]]] [[[1.,1.,1.]]] \
-i MobileNet-v2.onnx \
--output_signaturedefs \
--output_integer_quantized_tflite
   cp saved_model/MobileNet-v2_full_integer_quant.tflite MobileNet-v2-Quantized.tflite
fi
if [ -f MobileNet-v2-Quantized.tflite ]; then
   tflite_preprocess MobileNet-v2-Quantized.tflite  --scale 255
fi

if [ -f MobileNet-v2-Quantized.pre.tflite ]; then
    echo "Generating VNNX for V1000 ncomp configuration..."
    vnnx_compile -s V1000 -c ncomp -t MobileNet-v2-Quantized.pre.tflite  -o MobileNet-v2-Quantized_V1000_ncomp.vnnx
fi

if [ -f MobileNet-v2-Quantized_V1000_ncomp.vnnx ]; then
    echo "Running Simulation..."
    python $VBX_SDK/example/python/classifier.py MobileNet-v2-Quantized_V1000_ncomp.vnnx $VBX_SDK/tutorials/test_images/oreo.jpg 
    echo "C Simulation Command:"
    echo '$VBX_SDK/example/sim-c/sim-run-model MobileNet-v2-Quantized_V1000_ncomp.vnnx $VBX_SDK/tutorials/test_images/oreo.jpg CLASSIFY'
fi

deactivate
