
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
if [ ! -f $VBX_SDK/tutorials/imagenetv2_rgb_norm_20x224x224x3.npy ]; then
    generate_npy $VBX_SDK/tutorials/imagenetv2_rgb_20x224x224x3.npy -o $VBX_SDK/tutorials/imagenetv2_rgb_norm_20x224x224x3.npy -s 224 224  --norm 
fi

echo "Checking for onnx_resnet34-v1 files..."
if [ ! -f onnx_resnet34-v1.tflite ]; then
# model details @ https://github.com/onnx/models/tree/main/validated/vision/classification/resnet
wget -q --no-check-certificate https://media.githubusercontent.com/media/onnx/models/main/validated/vision/classification/resnet/model/resnet34-v1-7.onnx
    if [ ! -f calibration_image_sample_data_20x128x128x3_float32.npy ]; then
        wget -q --no-check-certificate https://github.com/Microchip-Vectorblox/assets/raw/refs/heads/main/npy_files/calibration_image_sample_data_20x128x128x3_float32.npy
    fi
fi

if [ -f resnet34-v1-7.onnx ]; then
   if ! echo "319090d33bcd78e8283b55984e71d741 resnet34-v1-7.onnx" | md5sum -c; then
       echo -e "\n There is an issue with the onnx_resnet34-v1 model file as the expected checksum does not match.\n The model source can be found at: https://github.com/onnx/models/tree/main/validated/vision/classification/resnet.\n If the model information has changed, please update this script and re-run the tutorial."
       exit 1
   fi
fi


# onnx2tf is an external model conversion tool to convert an onnx model to int8 tflite
# specific operation information can be found here: https://pypi.org/project/onnx2tf/
#  Purpose: Convert source model to int8 tflite format
#  - Required Inputs: onnx compliant model, calibration npy array
#  - Output: int8 tflite model
if [ ! -f onnx_resnet34-v1.tflite ]; then
   echo "Running ONNX2TF..."
   onnx2tf -cind data $VBX_SDK/tutorials/imagenetv2_rgb_norm_20x224x224x3.npy [[[[0.485,0.456,0.406]]]] [[[[0.229,0.224,0.225]]]] \
-i resnet34-v1-7.onnx \
--output_signaturedefs \
--output_integer_quantized_tflite
   cp saved_model/resnet34-v1-7_full_integer_quant.tflite onnx_resnet34-v1.tflite
fi

# tflite_preprocess is an internal tool used to add a preprocess layer to the start of the model
#  Purpose: adds a preprocess layer to the start of the model (if none, will just preprocess by adding a uint8->int8 layer)
#  - Required Inputs: tflite source model, additional arguments 
#  - Outputs: preprocessed tflite model
if [ -f onnx_resnet34-v1.tflite ]; then
   tflite_preprocess onnx_resnet34-v1.tflite  --mean 123.675 116.28 103.53 --scale 58.4 57.1 57.38
fi


# vnnx_compile is an internal tool that converts an int8 tflite file to a binary file that can be run on the SDK and VectorBlox FPGA
#  Purpose: converts int8 tflite to binary
#  - Required Inputs: int8 tflite, size configuration, compression configuration, output file name
#  - Outputs: binary object files(.hex and binary file)
if [ -f onnx_resnet34-v1.pre.tflite ]; then
    echo "Generating VNNX for V1000 ncomp configuration..."
    vnnx_compile -s V1000 -c ncomp -t onnx_resnet34-v1.pre.tflite  -o onnx_resnet34-v1_V1000_ncomp.vnnx
fi


# This step runs the final compiled binary in Python, it also shows how to run the same file in C simulation for SDK
#   *Currently C simulation is not supported for unstructured compression
if [ -f onnx_resnet34-v1_V1000_ncomp.vnnx ]; then
    echo "Running Simulation..."
    python $VBX_SDK/example/python/classifier.py onnx_resnet34-v1_V1000_ncomp.vnnx $VBX_SDK/tutorials/test_images/oreo.jpg 
    echo "C Simulation Command:"
    echo '$VBX_SDK/example/sim-c/sim-run-model onnx_resnet34-v1_V1000_ncomp.vnnx $VBX_SDK/tutorials/test_images/oreo.jpg CLASSIFY'
fi

deactivate
