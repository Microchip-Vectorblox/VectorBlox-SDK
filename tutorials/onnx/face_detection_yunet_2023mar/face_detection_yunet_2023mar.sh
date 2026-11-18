
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
if [ ! -f $VBX_SDK/tutorials/face_rgb_20x288x512x3.npy ]; then
    generate_npy $VBX_SDK/tutorials/face_rgb_20x273x481x3.npy -o $VBX_SDK/tutorials/face_rgb_20x288x512x3.npy -s 288 512 
fi

echo "Checking for face_detection_yunet_2023mar files..."

# model details @ https://github.com/yakhyo/face-recognition
if [ ! -f face_detection_yunet_2023mar.onnx ]; then
   wget -q --no-check-certificate https://github.com/opencv/opencv_zoo/raw/refs/tags/4.9.0/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
fi

    if [ ! -f calibration_image_sample_data_20x128x128x3_float32.npy ]; then
        wget -q --no-check-certificate https://github.com/Microchip-Vectorblox/assets/raw/refs/heads/main/npy_files/calibration_image_sample_data_20x128x128x3_float32.npy
    fi



# onnx2tf is an external model conversion tool to convert an onnx model to int8 tflite
# specific operation information can be found here: https://pypi.org/project/onnx2tf/
#  Purpose: Convert source model to int8 tflite format
#  - Required Inputs: onnx compliant model, calibration npy array
#  - Output: int8 tflite model
if [ ! -f face_detection_yunet_2023mar.tflite ]; then
   echo "Running ONNX2TF..."
   onnx2tf -cind input $VBX_SDK/tutorials/face_rgb_20x288x512x3.npy [[[[0.5,0.5,0.5]]]] [[[[0.502,0.502,0.502]]]]  \
--overwrite_input_shape input:1,3,288,512 \
--output_names_to_interrupt_model_conversion "270" "258" "264" "276" "260" "272" "266" "278" "262" "268" "280" "274" \
-i face_detection_yunet_2023mar.onnx \
--output_signaturedefs \
--output_integer_quantized_tflite
   cp saved_model/face_detection_yunet_2023mar_full_integer_quant.tflite face_detection_yunet_2023mar.tflite
fi

# tflite_preprocess is an internal tool used to add a preprocess layer to the start of the model
#  Purpose: adds a preprocess layer to the start of the model (if none, will just preprocess by adding a uint8->int8 layer)
#  - Required Inputs: tflite source model, additional arguments 
#  - Outputs: preprocessed tflite model
if [ -f face_detection_yunet_2023mar.tflite ]; then
   tflite_preprocess face_detection_yunet_2023mar.tflite   
fi


# vnnx_compile is an internal tool that converts an int8 tflite file to a binary file that can be run on the SDK and VectorBlox FPGA
#  Purpose: converts int8 tflite to binary
#  - Required Inputs: int8 tflite, size configuration, compression configuration, output file name
#  - Outputs: binary object files(.hex and binary file)
if [ -f face_detection_yunet_2023mar.pre.tflite ]; then
    echo "Generating VNNX for V1000 ncomp configuration..."
    vnnx_compile -s V1000 -c ncomp -t face_detection_yunet_2023mar.pre.tflite  -o face_detection_yunet_2023mar_V1000_ncomp.vnnx
fi


# This step runs the final compiled binary in Python, it also shows how to run the same file in C simulation for SDK
#   *Currently C simulation is not supported for unstructured compression
if [ -f face_detection_yunet_2023mar_V1000_ncomp.vnnx ]; then
    echo "Running Simulation..."
    python $VBX_SDK/example/python/yunetInfer.py face_detection_yunet_2023mar_V1000_ncomp.vnnx $VBX_SDK/tutorials/test_images/garden.jpg 
    echo "C Simulation Command:"
    echo '$VBX_SDK/example/sim-c/sim-run-model face_detection_yunet_2023mar_V1000_ncomp.vnnx $VBX_SDK/tutorials/test_images/garden.jpg YUNET'
fi

deactivate
