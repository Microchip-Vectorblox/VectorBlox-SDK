
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
if [ ! -f $VBX_SDK/tutorials/imagenetv2_20x128x128x3.npy ]; then
    generate_npy $VBX_SDK/tutorials/imagenetv2_rgb_20x224x224x3.npy -o $VBX_SDK/tutorials/imagenetv2_20x128x128x3.npy -s 128 128  -b 
fi

echo "Checking for mobilenet-v1-0.25-128 files..."

# model details @ https://github.com/openvinotoolkit/open_model_zoo/tree/2021.4.2/models/public/mobilenet-v1-0.25-128/
if [ ! -f mobilenet-v1-0.25-128.tflite ]; then 
omz_downloader --name mobilenet-v1-0.25-128
fi


if [ -f public/mobilenet-v1-0.25-128/mobilenet_v1_0.25_128_frozen.pb ]; then
   if ! echo "0fdca2e0ef5a760143da81bca3bb504f public/mobilenet-v1-0.25-128/mobilenet_v1_0.25_128_frozen.pb" | md5sum -c; then
       echo -e "\n There is an issue with the mobilenet-v1-0.25-128 model file as the expected checksum does not match.\n The model source can be found at: https://github.com/openvinotoolkit/open_model_zoo/tree/2021.4.2/models/public/mobilenet-v1-0.25-128/.\n If the model information has changed, please update this script and re-run the tutorial."
       exit 1
   fi
fi

if [ ! -f mobilenet-v1-0.25-128.tflite ]; then
   echo "Running Model Optimizer..."
   mo --input_model public/mobilenet-v1-0.25-128/mobilenet_v1_0.25_128_frozen.pb \
--input_shape [1,128,128,3] \
--mean_values [127.5,127.5,127.5] \
--reverse_input_channels \
--scale_values [127.5] \
--static_shape
fi

# openvino2tensorflow is an external model conversion tool to convert an openvino model to int8 tflite
# specific operation information can be found here: https://pypi.org/project/openvino2tensorflow
#  Purpose: Convert source model to int8 tflite format
#  - Required Inputs: openvino compliant model, calibration npy array
#  - Output: int8 tflite model
if [ ! -f mobilenet-v1-0.25-128.tflite ]; then
   echo "Running OpenVINO2Tensorflow..."
   openvino2tensorflow --load_dest_file_path_for_the_calib_npy $VBX_SDK/tutorials/imagenetv2_20x128x128x3.npy \
--keep_input_tensor_in_nchw \
--weight_replacement_config fix.json \
--model_path mobilenet_v1_0.25_128_frozen.xml \
--output_full_integer_quant_tflite \
--string_formulas_for_normalization '(data - [0.,0.,0.]) / [1.,1.,1.]'
   cp saved_model/model_full_integer_quant.tflite mobilenet-v1-0.25-128.tflite
fi


# tflite_preprocess is an internal tool used to add a preprocess layer to the start of the model
#  Purpose: adds a preprocess layer to the start of the model (if none, will just preprocess by adding a uint8->int8 layer)
#  - Required Inputs: tflite source model, additional arguments 
#  - Outputs: preprocessed tflite model
if [ -f mobilenet-v1-0.25-128.tflite ]; then
   tflite_preprocess mobilenet-v1-0.25-128.tflite   
fi


# vnnx_compile is an internal tool that converts an int8 tflite file to a binary file that can be run on the SDK and VectorBlox FPGA
#  Purpose: converts int8 tflite to binary
#  - Required Inputs: int8 tflite, size configuration, compression configuration, output file name
#  - Outputs: binary object files(.hex and binary file)
if [ -f mobilenet-v1-0.25-128.pre.tflite ]; then
    echo "Generating VNNX for V1000 ncomp configuration..."
    vnnx_compile -s V1000 -c ncomp -t mobilenet-v1-0.25-128.pre.tflite  -o mobilenet-v1-0.25-128_V1000_ncomp.vnnx
fi


# This step runs the final compiled binary in Python, it also shows how to run the same file in C simulation for SDK
#   *Currently C simulation is not supported for unstructured compression
if [ -f mobilenet-v1-0.25-128_V1000_ncomp.vnnx ]; then
    echo "Running Simulation..."
    python $VBX_SDK/example/python/classifier.py mobilenet-v1-0.25-128_V1000_ncomp.vnnx $VBX_SDK/tutorials/test_images/oreo.jpg 
    echo "C Simulation Command:"
    echo '$VBX_SDK/example/sim-c/sim-run-model mobilenet-v1-0.25-128_V1000_ncomp.vnnx $VBX_SDK/tutorials/test_images/oreo.jpg CLASSIFY'
fi

deactivate
