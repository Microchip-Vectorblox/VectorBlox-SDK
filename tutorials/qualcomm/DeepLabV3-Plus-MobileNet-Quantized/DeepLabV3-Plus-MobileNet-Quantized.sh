
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

echo "Downloading DeepLabV3-Plus-MobileNet-Quantized..."
# model details @ https://aihub.qualcomm.com/mobile/models/deeplabv3_plus_mobilenet_quantized
if [ ! -f DeepLabV3-Plus-MobileNet-Quantized.tflite ]; then
    wget -q --no-check-certificate https://huggingface.co/qualcomm/DeepLabV3-Plus-MobileNet-Quantized/resolve/main/DeepLabV3-Plus-MobileNet-Quantized.tflite
fi

if [ -f DeepLabV3-Plus-MobileNet-Quantized.tflite ]; then
   tflite_preprocess DeepLabV3-Plus-MobileNet-Quantized.tflite  --scale 255
fi

if [ -f DeepLabV3-Plus-MobileNet-Quantized.pre.tflite ]; then
   tflite_postprocess DeepLabV3-Plus-MobileNet-Quantized.pre.tflite  --dataset VOC \
--opacity 0.8 \
--height 1080 \
--width 1920
fi

if [ -f DeepLabV3-Plus-MobileNet-Quantized.pre.post.tflite ]; then
    echo "Generating VNNX for V1000 configuration..."
    vnnx_compile -c V1000 -t DeepLabV3-Plus-MobileNet-Quantized.pre.post.tflite -o DeepLabV3-Plus-MobileNet-Quantized.vnnx
fi

if [ -f DeepLabV3-Plus-MobileNet-Quantized.vnnx ]; then
    echo "Running Simulation..."
    python $VBX_SDK/example/python/segmentation.py DeepLabV3-Plus-MobileNet-Quantized.vnnx $VBX_SDK/tutorials/test_images/A0PQ76.jpg --inj 
    echo "C Simulation Command:"
    echo '$VBX_SDK/example/sim-c/sim-run-model DeepLabV3-Plus-MobileNet-Quantized.vnnx $VBX_SDK/tutorials/test_images/A0PQ76.jpg  '
fi

deactivate
