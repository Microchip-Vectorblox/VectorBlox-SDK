
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


# Ultralytics YOLO models are available under the AGPL-3.0 open-source license.
# Projects that are not open source require an Ultralytics Enterprise License. To
# obtain a commercial license for R&D and production use without open-source obligations,
# please complete the licensing form at https://www.ultralytics.com/license.
    

set -e
echo "Checking and activating VBX Python Environment..."
if [ -z $VBX_SDK ]; then
    echo "\$VBX_SDK not set. Please run 'source setup_vars.sh' from the SDK's root folder" && exit 1
fi
source $VBX_SDK/vbx_env/bin/activate

echo "Checking for yolo26n_sym files..."

# model details @ https://github.com/ultralytics/ultralytics/
[ -f coco.names ] || wget -q https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names
if [ ! -f yolo26n_sym.tflite ]; then
    uv sync --group cpu
    uv run python yolo_tflite.py --model yolo26n
    tflite_clean -i yolo26n_aeq_softmax_add_int8.tflite
    tflite_cut yolo26n_aeq_softmax_add_int8.clean.tflite -c 1
    cp yolo26n_aeq_softmax_add_int8.clean.1.tflite yolo26n_sym.tflite
fi



# tflite_cut is an internal tool used to split an existing model into smaller models
#  Purpose: cuts a model into smaller subsections, can be used to decrease runtime or for debugging purposes
#  - Required Inputs: tflite source model, cut section(s)
#  - Outputs: tflite model
if [ -f yolo26n_sym.tflite ]; then 
   echo "Cutting graph" 
   tflite_cut yolo26n_sym.tflite -c 486 499 512 539 562 585
   mv yolo26n_sym.0.tflite yolo26n_sym.cut.tflite 
fi


# tflite_preprocess is an internal tool used to add a preprocess layer to the start of the model
#  Purpose: adds a preprocess layer to the start of the model (if none, will just preprocess by adding a uint8->int8 layer)
#  - Required Inputs: tflite source model, additional arguments 
#  - Outputs: preprocessed tflite model
if [ -f yolo26n_sym.cut.tflite ]; then
   tflite_preprocess yolo26n_sym.cut.tflite  --scale 255
fi


# tflite_transform is an internal tool used to add various custom transformations to the model
# additional information found here: https://github.com/Microchip-Vectorblox/VectorBlox-SDK/blob/master/python/vbx/vbx/generate/transform_tflite.py
# supported arguments are found in the `passes` struct at the top#  Purpose: Adds various transformations to a tflite model
#  - Required Inputs: tflite source model, additional arguments
#  - Outputs: transformed tflite model
if [ -f yolo26n_sym.cut.pre.tflite ]; then
   tflite_transform yolo26n_sym.cut.pre.tflite  -p YOLO_ATTN
fi


# vnnx_compile is an internal tool that converts an int8 tflite file to a binary file that can be run on the SDK and VectorBlox FPGA
#  Purpose: converts int8 tflite to binary
#  - Required Inputs: int8 tflite, size configuration, compression configuration, output file name
#  - Outputs: binary object files(.hex and binary file)
if [ -f yolo26n_sym.cut.pre.tr.tflite ]; then
    echo "Generating VNNX for V1000 ncomp configuration..."
    vnnx_compile -s V1000 -c ncomp -t yolo26n_sym.cut.pre.tr.tflite  -o yolo26n_sym_V1000_ncomp.vnnx
fi


# This step runs the final compiled binary in Python, it also shows how to run the same file in C simulation for SDK
#   *Currently C simulation is not supported for unstructured compression
if [ -f yolo26n_sym_V1000_ncomp.vnnx ]; then
    echo "Running Simulation..."
    python $VBX_SDK/example/python/yoloInfer.py yolo26n_sym_V1000_ncomp.vnnx $VBX_SDK/tutorials/test_images/dog.jpg -v 26 -l coco.names 
    echo "C Simulation Command:"
    echo '$VBX_SDK/example/sim-c/sim-run-model yolo26n_sym_V1000_ncomp.vnnx $VBX_SDK/tutorials/test_images/dog.jpg OBJECT_DETECT'
fi

deactivate
