
##########################################################
#  _    __          __             ____  __              #
# | |  / /__  _____/ /_____  _____/ __ )/ /___  _  __    #
# | | / / _ \/ ___/ __/ __ \/ ___/ __  / / __ \| |/_/    #
# | |/ /  __/ /__/ /_/ /_/ / /  / /_/ / / /_/ />  <      #
# |___/\___/\___/\__/\____/_/  /_____/_/\____/_/|_|      #
#                                                        #
# https://github.com/Microchip-Vectorblox/VectorBlox-SDK #
# v3.1                                                   #
#                                                        #
##########################################################



set -e
echo "Checking and activating VBX Python Environment..."
if [ -z $VBX_SDK ]; then
    echo "\$VBX_SDK not set. Please run 'source setup_vars.sh' from the SDK's root folder" && exit 1
fi
source $VBX_SDK/vbx_env/bin/activate

echo "Checking for Midas-V2-Quantized files..."

# model details @ https://aihub.qualcomm.com/models/midas
if [ ! -f Midas-V2-Quantized.tflite ]; then
    wget -q --no-check-certificate https://huggingface.co/qualcomm/Midas-V2/resolve/v0.32.0/Midas-V2_w8a8.tflite
    mv Midas-V2_w8a8.tflite Midas-V2-Quantized.tflite
fi


if [ -f Midas-V2-Quantized.tflite ]; then
   if ! echo "9c3b309a1c580fe55c60a8bbdfea27b7 Midas-V2-Quantized.tflite" | md5sum -c; then
       echo -e "\n There is an issue with the Midas-V2-Quantized model file as the expected checksum does not match.\n The model source can be found at: https://aihub.qualcomm.com/models/midas.\n If the model information has changed, please update this script and re-run the tutorial."
       exit 1
   fi
fi


# tflite_preprocess is an internal tool used to add a preprocess layer to the start of the model
#  Purpose: adds a preprocess layer to the start of the model (if none, will just preprocess by adding a uint8->int8 layer)
#  - Required Inputs: tflite source model, additional arguments 
#  - Outputs: preprocessed tflite model
if [ -f Midas-V2-Quantized.tflite ]; then
   tflite_preprocess Midas-V2-Quantized.tflite  --scale 255
fi


# tflite_postprocess is an internal tool that adds a postprocess layer near the end of the model
#  Purpose: adds a postprocess layer at the end of the model
#  - Required Inputs: tflite source model, additional postprocessing arguments
#  - Outputs: tflite model with postprocessed layer at end
if [ -f Midas-V2-Quantized.pre.tflite ]; then
   tflite_postprocess Midas-V2-Quantized.pre.tflite  --post-process-layer PIXEL_DEPTH \
--opacity 0.8 \
--height 1080 \
--width 1920
fi


# vnnx_compile is an internal tool that converts an int8 tflite file to a binary file that can be run on the SDK and VectorBlox FPGA
#  Purpose: converts int8 tflite to binary
#  - Required Inputs: int8 tflite, size configuration, compression configuration, output file name
#  - Outputs: binary object files(.hex and binary file)
if [ -f Midas-V2-Quantized.pre.post.tflite ]; then
    echo "Generating VNNX for V1000 ncomp configuration..."
    vnnx_compile -s V1000 -c ncomp -t Midas-V2-Quantized.pre.post.tflite  -o Midas-V2-Quantized_V1000_ncomp.vnnx
fi


# This step runs the final compiled binary in Python, it also shows how to run the same file in C simulation for SDK
#   *Currently C simulation is not supported for unstructured compression
if [ -f Midas-V2-Quantized_V1000_ncomp.vnnx ]; then
    echo "Running Simulation..."
    python $VBX_SDK/example/python/segmentation.py Midas-V2-Quantized_V1000_ncomp.vnnx $VBX_SDK/tutorials/test_images/A0PQ76.jpg --dataset depth --inj 
    echo "C Simulation Command:"
    echo '$VBX_SDK/example/sim-c/sim-run-model Midas-V2-Quantized_V1000_ncomp.vnnx $VBX_SDK/tutorials/test_images/A0PQ76.jpg  '
fi

deactivate
