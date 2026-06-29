
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
if [ ! -f $VBX_SDK/tutorials/imagenetv2_rgb_norm_100x299x299x3.npy ]; then
    generate_npy $VBX_SDK/tutorials/imagenetv2_rgb_20x224x224x3.npy -o $VBX_SDK/tutorials/imagenetv2_rgb_norm_100x299x299x3.npy -s 299 299  --norm 
fi

echo "Checking for torchvision_inception_v3 files..."
if [ ! -f torchvision_inception_v3.tflite ]; then
# model details @ https://pytorch.org/vision/0.14/models/inception.html
python $VBX_SDK/tutorials/torchvision_to_onnx.py inception_v3 -i 299
    if [ ! -f calibration_image_sample_data_20x128x128x3_float32.npy ]; then
        wget -q --no-check-certificate https://github.com/Microchip-Vectorblox/assets/raw/refs/heads/main/npy_files/calibration_image_sample_data_20x128x128x3_float32.npy
    fi
fi

if [ -f inception_v3.onnx ]; then
   if ! echo "a28b69fa0dda9fbeccc1c2d9f5b03d22 inception_v3.onnx" | md5sum -c; then
       echo -e "\n There is an issue with the torchvision_inception_v3 model file as the expected checksum does not match.\n The model source can be found at: https://pytorch.org/vision/0.14/models/inception.html.\n If the model information has changed, please update this script and re-run the tutorial."
       exit 1
   fi
fi


# onnx2tf is an external model conversion tool to convert an onnx model to int8 tflite
# specific operation information can be found here: https://pypi.org/project/onnx2tf/
#  Purpose: Convert source model to int8 tflite format
#  - Required Inputs: onnx compliant model, calibration npy array
#  - Output: int8 tflite model
if [ ! -f torchvision_inception_v3.tflite ]; then
   echo "Running ONNX2TF..."
   onnx2tf -cind input.1 $VBX_SDK/tutorials/imagenetv2_rgb_norm_100x299x299x3.npy [[[[0.485,0.456,0.406]]]] [[[[0.229,0.224,0.225]]]] \
-i inception_v3.onnx \
--output_signaturedefs \
--output_integer_quantized_tflite
   cp saved_model/inception_v3_full_integer_quant.tflite torchvision_inception_v3.tflite
fi

# tflite_preprocess is an internal tool used to add a preprocess layer to the start of the model
#  Purpose: adds a preprocess layer to the start of the model (if none, will just preprocess by adding a uint8->int8 layer)
#  - Required Inputs: tflite source model, additional arguments 
#  - Outputs: preprocessed tflite model
if [ -f torchvision_inception_v3.tflite ]; then
   tflite_preprocess torchvision_inception_v3.tflite  --mean 123.675 116.28 103.53 --scale 58.4 57.1 57.38
fi


# vnnx_compile is an internal tool that converts an int8 tflite file to a binary file that can be run on the SDK and VectorBlox FPGA
#  Purpose: converts int8 tflite to binary
#  - Required Inputs: int8 tflite, size configuration, compression configuration, output file name
#  - Outputs: binary object files(.hex and binary file)
if [ -f torchvision_inception_v3.pre.tflite ]; then
    echo "Generating VNNX for V1000 ncomp configuration..."
    vnnx_compile -s V1000 -c ncomp -t torchvision_inception_v3.pre.tflite  -o torchvision_inception_v3_V1000_ncomp.vnnx
fi


# This step runs the final compiled binary in Python, it also shows how to run the same file in C simulation for SDK
#   *Currently C simulation is not supported for unstructured compression
if [ -f torchvision_inception_v3_V1000_ncomp.vnnx ]; then
    echo "Running Simulation..."
    python $VBX_SDK/example/python/classifier.py torchvision_inception_v3_V1000_ncomp.vnnx $VBX_SDK/tutorials/test_images/oreo.jpg 
    echo "C Simulation Command:"
    echo '$VBX_SDK/example/sim-c/sim-run-model torchvision_inception_v3_V1000_ncomp.vnnx $VBX_SDK/tutorials/test_images/oreo.jpg CLASSIFY'
fi

deactivate
