
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

echo "Downloading Midas-V2-Quantized..."
# model details @ https://aihub.qualcomm.com/models/midas
if [ ! -f Midas-V2-Quantized.tflite ]; then
    wget -q --no-check-certificate https://huggingface.co/qualcomm/Midas-V2-Quantized/resolve/main/Midas-V2-Quantized.tflite
fi

if [ -f Midas-V2-Quantized.tflite ]; then
   tflite_preprocess Midas-V2-Quantized.tflite  --scale 255
fi

if [ -f Midas-V2-Quantized.pre.tflite ]; then
   tflite_postprocess Midas-V2-Quantized.pre.tflite  --dataset depth \
--opacity 0.8 \
--height 1080 \
--width 1920
fi

if [ -f Midas-V2-Quantized.pre.post.tflite ]; then
    echo "Generating VNNX for V1000 configuration..."
    vnnx_compile -c V1000 -t Midas-V2-Quantized.pre.post.tflite -o Midas-V2-Quantized.vnnx
fi

if [ -f Midas-V2-Quantized.vnnx ]; then
    echo "Running Simulation..."
    python $VBX_SDK/example/python/segmentation.py Midas-V2-Quantized.vnnx $VBX_SDK/tutorials/test_images/A0PQ76.jpg --dataset depth --inj 
    echo "C Simulation Command:"
    echo '$VBX_SDK/example/sim-c/sim-run-model Midas-V2-Quantized.vnnx $VBX_SDK/tutorials/test_images/A0PQ76.jpg  '
fi

deactivate
