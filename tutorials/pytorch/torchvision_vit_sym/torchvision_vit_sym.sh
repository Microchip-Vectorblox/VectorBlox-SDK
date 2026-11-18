
##########################################################
#  _    __          __             ____  __              #
# | |  / /__  _____/ /_____  _____/ __ )/ /___  _  __    #
# | | / / _ \/ ___/ __/ __ \/ ___/ __  / / __ \| |/_/    #
# | |/ /  __/ /__/ /_/ /_/ / /  / /_/ / / /_/ />  <      #
# |___/\___/\___/\__/\____/_/  /_____/_/\____/_/|_|      #
#                                                        #
# https://github.com/Microchip-Vectorblox/VectorBlox-SDK #
# v3.2                                                   #
#                                                        #
##########################################################



set -e
echo "Checking and activating VBX Python Environment..."
if [ -z $VBX_SDK ]; then
    echo "\$VBX_SDK not set. Please run 'source setup_vars.sh' from the SDK's root folder" && exit 1
fi
source $VBX_SDK/vbx_env/bin/activate

echo "Checking for torchvision_vit_sym files..."

# model details @ https://huggingface.co/qualcomm/VIT
if [ ! -f torchvision_vit_sym.tflite ]; then
    uv sync --group cpu
    uv run python vit_tflite.py
    tflite_clean -i vit_b_16_aeq_softmax_add_attention_fc_add_bmm2_output_no_quantize_int8.tflite
    tflite_cut vit_b_16_aeq_softmax_add_attention_fc_add_bmm2_output_no_quantize_int8.clean.tflite -c 0
    cp vit_b_16_aeq_softmax_add_attention_fc_add_bmm2_output_no_quantize_int8.clean.1.tflite torchvision_vit_sym.tflite
fi



# tflite_preprocess is an internal tool used to add a preprocess layer to the start of the model
#  Purpose: adds a preprocess layer to the start of the model (if none, will just preprocess by adding a uint8->int8 layer)
#  - Required Inputs: tflite source model, additional arguments 
#  - Outputs: preprocessed tflite model
if [ -f torchvision_vit_sym.tflite ]; then
   tflite_preprocess torchvision_vit_sym.tflite  --scale 255
fi


# tflite_transform is an internal tool used to add various custom transformations to the model
# additional information found here: https://github.com/Microchip-Vectorblox/VectorBlox-SDK/blob/master/python/vbx/vbx/generate/transform_tflite.py
# supported arguments are found in the `passes` struct at the top#  Purpose: Adds various transformations to a tflite model
#  - Required Inputs: tflite source model, additional arguments
#  - Outputs: transformed tflite model
if [ -f torchvision_vit_sym.pre.tflite ]; then
   tflite_transform torchvision_vit_sym.pre.tflite  -p FRAC_BATCH FUSE_ADJ FUSE_SLICES UNPACK_SLICE
fi


# vnnx_compile is an internal tool that converts an int8 tflite file to a binary file that can be run on the SDK and VectorBlox FPGA
#  Purpose: converts int8 tflite to binary
#  - Required Inputs: int8 tflite, size configuration, compression configuration, output file name
#  - Outputs: binary object files(.hex and binary file)
if [ -f torchvision_vit_sym.pre.tr.tflite ]; then
    echo "Generating VNNX for V1000 ncomp configuration..."
    vnnx_compile -s V1000 -c ncomp -t torchvision_vit_sym.pre.tr.tflite  -o torchvision_vit_sym_V1000_ncomp.vnnx
fi


# This step runs the final compiled binary in Python, it also shows how to run the same file in C simulation for SDK
#   *Currently C simulation is not supported for unstructured compression
if [ -f torchvision_vit_sym_V1000_ncomp.vnnx ]; then
    echo "Running Simulation..."
    python $VBX_SDK/example/python/classifier.py torchvision_vit_sym_V1000_ncomp.vnnx $VBX_SDK/tutorials/test_images/oreo.jpg 
    echo "C Simulation Command:"
    echo '$VBX_SDK/example/sim-c/sim-run-model torchvision_vit_sym_V1000_ncomp.vnnx $VBX_SDK/tutorials/test_images/oreo.jpg CLASSIFY'
fi

deactivate
