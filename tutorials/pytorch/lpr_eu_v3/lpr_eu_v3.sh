
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


# generate_npy is an internal tool that creates a npy array
#  Purpose: Generates a npy array if an existing one does not exist, this is using custom img data
#  - Required Inputs: source dataset, output name, size
#  - Output: npy array
echo "Checking for Numpy calibration data file..."
if [ ! -f $VBX_SDK/tutorials/sample_plates_20x34x146x3.npy ]; then
    generate_npy $VBX_SDK/tutorials/sample_plates_20x34x146x3.npy -o $VBX_SDK/tutorials/sample_plates_20x34x146x3.npy -s 34 146  -b 
fi

echo "Checking for lpr_eu_v3 files..."

# model details @ pytorch/lpr_eu_v3/README.md
if [ ! -f lpr_eu_v3.tflite ]; then
   [ -f lpr_eu_v3.onnx ] || wget -q --no-check-certificate https://github.com/Microchip-Vectorblox/assets/releases/download/assets/lpr_eu_v3.onnx
fi


if [ -f lpr_eu_v3.onnx ]; then
   if ! echo "d462388d2501b21072424fed7047051f lpr_eu_v3.onnx" | md5sum -c; then
       echo -e "\n There is an issue with the lpr_eu_v3 model file as the expected checksum does not match.\n The model source can be found at: pytorch/lpr_eu_v3/README.md.\n If the model information has changed, please update this script and re-run the tutorial."
       exit 1
   fi
fi

if [ ! -f lpr_eu_v3.tflite ]; then
   echo "Running Model Optimizer..."
   mo --input_model lpr_eu_v3.onnx \
--scale_values [255.] \
--static_shape \
--input_shape [1,3,34,146]
fi

# openvino2tensorflow is an external model conversion tool to convert an openvino model to int8 tflite
# specific operation information can be found here: https://pypi.org/project/openvino2tensorflow
#  Purpose: Convert source model to int8 tflite format
#  - Required Inputs: openvino compliant model, calibration npy array
#  - Output: int8 tflite model
if [ ! -f lpr_eu_v3.tflite ]; then
   echo "Running OpenVINO2Tensorflow..."
   openvino2tensorflow --load_dest_file_path_for_the_calib_npy $VBX_SDK/tutorials/sample_plates_20x34x146x3.npy \
--model_path lpr_eu_v3.xml \
--output_full_integer_quant_tflite \
--string_formulas_for_normalization '(data - [0.,0.,0.]) / [1.,1.,1.]'
   cp saved_model/model_full_integer_quant.tflite lpr_eu_v3.tflite
fi


# tflite_preprocess is an internal tool used to add a preprocess layer to the start of the model
#  Purpose: adds a preprocess layer to the start of the model (if none, will just preprocess by adding a uint8->int8 layer)
#  - Required Inputs: tflite source model, additional arguments 
#  - Outputs: preprocessed tflite model
if [ -f lpr_eu_v3.tflite ]; then
   tflite_preprocess lpr_eu_v3.tflite   
fi


# vnnx_compile is an internal tool that converts an int8 tflite file to a binary file that can be run on the SDK and VectorBlox FPGA
#  Purpose: converts int8 tflite to binary
#  - Required Inputs: int8 tflite, size configuration, compression configuration, output file name
#  - Outputs: binary object files(.hex and binary file)
if [ -f lpr_eu_v3.pre.tflite ]; then
    echo "Generating VNNX for V1000 ncomp configuration..."
    vnnx_compile -s V1000 -c ncomp -t lpr_eu_v3.pre.tflite  -o lpr_eu_v3_V1000_ncomp.vnnx
fi


# This step runs the final compiled binary in Python, it also shows how to run the same file in C simulation for SDK
#   *Currently C simulation is not supported for unstructured compression
if [ -f lpr_eu_v3_V1000_ncomp.vnnx ]; then
    echo "Running Simulation..."
    python $VBX_SDK/example/python/recognize_plate_eu.py lpr_eu_v3_V1000_ncomp.vnnx $VBX_SDK/tutorials/test_images/A358CC82.jpg 
    echo "C Simulation Command:"
    echo '$VBX_SDK/example/sim-c/sim-run-model lpr_eu_v3_V1000_ncomp.vnnx $VBX_SDK/tutorials/test_images/A358CC82.jpg LPR'
fi

deactivate
