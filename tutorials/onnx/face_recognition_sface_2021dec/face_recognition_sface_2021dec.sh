
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


# generate_npy is an internal tool that creates a npy array
#  Purpose: Generates a npy array if an existing one does not exist, this is using custom img data
#  - Required Inputs: source dataset, output name, size
#  - Output: npy array
echo "Checking for Numpy calibration data file..."
if [ ! -f $VBX_SDK/tutorials/imagenetv2_20x112x112x3.npy ]; then
    generate_npy $VBX_SDK/tutorials/imagenetv2_rgb_20x224x224x3.npy -o $VBX_SDK/tutorials/imagenetv2_20x112x112x3.npy -s 112 112  -b 
fi

echo "Checking for face_recognition_sface_2021dec files..."

# model details @ https://github.com/yakhyo/face-recognition
if [ ! -f face_recognition_sface_2021dec.tflite ]; then
   wget -q --no-check-certificate https://github.com/opencv/opencv_zoo/raw/refs/heads/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx
fi

    if [ ! -f calibration_image_sample_data_20x128x128x3_float32.npy ]; then
        wget -q --no-check-certificate https://github.com/Microchip-Vectorblox/assets/raw/refs/heads/main/npy_files/calibration_image_sample_data_20x128x128x3_float32.npy
    fi



# onnx2tf is an external model conversion tool to convert an onnx model to int8 tflite
# specific operation information can be found here: https://pypi.org/project/onnx2tf/
#  Purpose: Convert source model to int8 tflite format
#  - Required Inputs: onnx compliant model, calibration npy array
#  - Output: int8 tflite model
if [ ! -f face_recognition_sface_2021dec.tflite ]; then
   echo "Running ONNX2TF..."
   onnx2tf -cind data $VBX_SDK/tutorials/imagenetv2_20x112x112x3.npy  [[[[0,0,0]]]] [[[[1,1,1]]]]  \
-i face_recognition_sface_2021dec.onnx \
--output_signaturedefs \
--output_integer_quantized_tflite
   cp saved_model/face_recognition_sface_2021dec_full_integer_quant.tflite face_recognition_sface_2021dec.tflite
fi

# tflite_preprocess is an internal tool used to add a preprocess layer to the start of the model
#  Purpose: adds a preprocess layer to the start of the model (if none, will just preprocess by adding a uint8->int8 layer)
#  - Required Inputs: tflite source model, additional arguments 
#  - Outputs: preprocessed tflite model
if [ -f face_recognition_sface_2021dec.tflite ]; then
   tflite_preprocess face_recognition_sface_2021dec.tflite  --scale 1
fi


# vnnx_compile is an internal tool that converts an int8 tflite file to a binary file that can be run on the SDK and VectorBlox FPGA
#  Purpose: converts int8 tflite to binary
#  - Required Inputs: int8 tflite, size configuration, compression configuration, output file name
#  - Outputs: binary object files(.hex and binary file)
if [ -f face_recognition_sface_2021dec.pre.tflite ]; then
    echo "Generating VNNX for V1000 ncomp configuration..."
    vnnx_compile -s V1000 -c ncomp -t face_recognition_sface_2021dec.pre.tflite  -o face_recognition_sface_2021dec_V1000_ncomp.vnnx
fi


# This step runs the final compiled binary in Python, it also shows how to run the same file in C simulation for SDK
#   *Currently C simulation is not supported for unstructured compression
if [ -f face_recognition_sface_2021dec_V1000_ncomp.vnnx ]; then
    echo "Running Simulation..."
    python $VBX_SDK/example/python/face_compare.py face_recognition_sface_2021dec_V1000_ncomp.vnnx $VBX_SDK/tutorials/test_images/MattDamon0001_arcface.jpg $VBX_SDK/tutorials/test_images/MattDamon0002_arcface.jpg 
    echo "C Simulation Command:"
    echo '$VBX_SDK/example/sim-c/sim-run-model face_recognition_sface_2021dec_V1000_ncomp.vnnx $VBX_SDK/tutorials/test_images/MattDamon0001_arcface.jpg FACEMCP'
fi

deactivate
