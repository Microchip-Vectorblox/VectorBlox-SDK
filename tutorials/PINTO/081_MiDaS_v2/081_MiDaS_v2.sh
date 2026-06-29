
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

echo "Checking for 081_MiDaS_v2 files..."

# model details @ https://github.com/PINTO0309/PINTO_model_zoo
if ! [ -f 081_MiDaS_v2.tflite ]; then
    mkdir -p temp
    cd temp
    wget -q https://raw.githubusercontent.com/PINTO0309/PINTO_model_zoo/refs/heads/main/081_MiDaS_v2/download.sh
    bash download.sh
    cd ..
    cp temp/tflite_from_saved_model/model_full_integer_quant.tflite 081_MiDaS_v2.tflite
    rm -rf temp
fi


if [ -f 081_MiDaS_v2.tflite ]; then
   if ! echo "a37e2ec3d0c28b6283adddb46c8923c1 081_MiDaS_v2.tflite" | md5sum -c; then
       echo -e "\n There is an issue with the 081_MiDaS_v2 model file as the expected checksum does not match.\n The model source can be found at: https://github.com/PINTO0309/PINTO_model_zoo.\n If the model information has changed, please update this script and re-run the tutorial."
       exit 1
   fi
fi


# tflite_cut is an internal tool used to split an existing model into smaller models
#  Purpose: cuts a model into smaller subsections, can be used to decrease runtime or for debugging purposes
#  - Required Inputs: tflite source model, cut section(s)
#  - Outputs: tflite model
if [ -f 081_MiDaS_v2.tflite ]; then 
   echo "Cutting graph" 
   tflite_cut 081_MiDaS_v2.tflite -c 137
   mv 081_MiDaS_v2.0.tflite 081_MiDaS_v2.cut.tflite 
fi


# tflite_preprocess is an internal tool used to add a preprocess layer to the start of the model
#  Purpose: adds a preprocess layer to the start of the model (if none, will just preprocess by adding a uint8->int8 layer)
#  - Required Inputs: tflite source model, additional arguments 
#  - Outputs: preprocessed tflite model
if [ -f 081_MiDaS_v2.cut.tflite ]; then
   tflite_preprocess 081_MiDaS_v2.cut.tflite   
fi


# tflite_postprocess is an internal tool that adds a postprocess layer near the end of the model
#  Purpose: adds a postprocess layer at the end of the model
#  - Required Inputs: tflite source model, additional postprocessing arguments
#  - Outputs: tflite model with postprocessed layer at end
if [ -f 081_MiDaS_v2.cut.pre.tflite ]; then
   tflite_postprocess 081_MiDaS_v2.cut.pre.tflite  --post-process-layer PIXEL_DEPTH \
--opacity 0.8 \
--height 1080 \
--width 1920
fi


# vnnx_compile is an internal tool that converts an int8 tflite file to a binary file that can be run on the SDK and VectorBlox FPGA
#  Purpose: converts int8 tflite to binary
#  - Required Inputs: int8 tflite, size configuration, compression configuration, output file name
#  - Outputs: binary object files(.hex and binary file)
if [ -f 081_MiDaS_v2.cut.pre.post.tflite ]; then
    echo "Generating VNNX for V1000 ncomp configuration..."
    vnnx_compile -s V1000 -c ncomp -t 081_MiDaS_v2.cut.pre.post.tflite  -o 081_MiDaS_v2_V1000_ncomp.vnnx
fi


# This step runs the final compiled binary in Python, it also shows how to run the same file in C simulation for SDK
#   *Currently C simulation is not supported for unstructured compression
if [ -f 081_MiDaS_v2_V1000_ncomp.vnnx ]; then
    echo "Running Simulation..."
    python $VBX_SDK/example/python/segmentation.py 081_MiDaS_v2_V1000_ncomp.vnnx $VBX_SDK/tutorials/test_images/A0PQ76.jpg --dataset depth 
    echo "C Simulation Command:"
    echo '$VBX_SDK/example/sim-c/sim-run-model 081_MiDaS_v2_V1000_ncomp.vnnx $VBX_SDK/tutorials/test_images/A0PQ76.jpg  '
fi

deactivate
