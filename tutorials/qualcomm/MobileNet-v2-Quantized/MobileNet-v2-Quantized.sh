
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
if [ ! -f $VBX_SDK/tutorials/coco2017_rgb_norm_20x224x224x3.npy ]; then
    generate_npy $VBX_SDK/tutorials/coco2017_rgb_20x416x416x3.npy -o $VBX_SDK/tutorials/coco2017_rgb_norm_20x224x224x3.npy -s 224 224  --norm 
fi

echo "Checking for MobileNet-v2-Quantized files..."

# model details @ https://aihub.qualcomm.com/models/mobilenet_v2
if [ ! -f MobileNet-v2-Quantized.tflite ]; then
    wget --no-check-certificate https://huggingface.co/qualcomm/MobileNet-v2/resolve/v0.32.0/MobileNet-v2.onnx
fi
    if [ ! -f calibration_image_sample_data_20x128x128x3_float32.npy ]; then
        wget -q --no-check-certificate https://github.com/Microchip-Vectorblox/assets/raw/refs/heads/main/npy_files/calibration_image_sample_data_20x128x128x3_float32.npy
    fi


if [ -f MobileNet-v2.onnx ]; then
   if ! echo "f83f6569aa99014a8267e14680cd6b7a MobileNet-v2.onnx" | md5sum -c; then
       echo -e "\n There is an issue with the MobileNet-v2-Quantized model file as the expected checksum does not match.\n The model source can be found at: https://aihub.qualcomm.com/models/mobilenet_v2.\n If the model information has changed, please update this script and re-run the tutorial."
       exit 1
   fi
fi


# onnx2tf is an external model conversion tool to convert an onnx model to int8 tflite
# specific operation information can be found here: https://pypi.org/project/onnx2tf/
#  Purpose: Convert source model to int8 tflite format
#  - Required Inputs: onnx compliant model, calibration npy array
#  - Output: int8 tflite model
if [ ! -f MobileNet-v2-Quantized.tflite ]; then
   echo "Running ONNX2TF..."
   onnx2tf -cind image_tensor $VBX_SDK/tutorials/coco2017_rgb_norm_20x224x224x3.npy [[[0.,0.,0.]]] [[[1.,1.,1.]]] \
-i MobileNet-v2.onnx \
--output_signaturedefs \
--output_integer_quantized_tflite
   cp saved_model/MobileNet-v2_full_integer_quant.tflite MobileNet-v2-Quantized.tflite
fi

# tflite_preprocess is an internal tool used to add a preprocess layer to the start of the model
#  Purpose: adds a preprocess layer to the start of the model (if none, will just preprocess by adding a uint8->int8 layer)
#  - Required Inputs: tflite source model, additional arguments 
#  - Outputs: preprocessed tflite model
if [ -f MobileNet-v2-Quantized.tflite ]; then
   tflite_preprocess MobileNet-v2-Quantized.tflite  --scale 255
fi


# vnnx_compile is an internal tool that converts an int8 tflite file to a binary file that can be run on the SDK and VectorBlox FPGA
#  Purpose: converts int8 tflite to binary
#  - Required Inputs: int8 tflite, size configuration, compression configuration, output file name
#  - Outputs: binary object files(.hex and binary file)
if [ -f MobileNet-v2-Quantized.pre.tflite ]; then
    echo "Generating VNNX for V1000 ncomp configuration..."
    vnnx_compile -s V1000 -c ncomp -t MobileNet-v2-Quantized.pre.tflite  -o MobileNet-v2-Quantized_V1000_ncomp.vnnx
fi


# This step runs the final compiled binary in Python, it also shows how to run the same file in C simulation for SDK
#   *Currently C simulation is not supported for unstructured compression
if [ -f MobileNet-v2-Quantized_V1000_ncomp.vnnx ]; then
    echo "Running Simulation..."
    python $VBX_SDK/example/python/classifier.py MobileNet-v2-Quantized_V1000_ncomp.vnnx $VBX_SDK/tutorials/test_images/oreo.jpg 
    echo "C Simulation Command:"
    echo '$VBX_SDK/example/sim-c/sim-run-model MobileNet-v2-Quantized_V1000_ncomp.vnnx $VBX_SDK/tutorials/test_images/oreo.jpg CLASSIFY'
fi

deactivate
