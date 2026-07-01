
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
if [ ! -f $VBX_SDK/tutorials/coco2017_rgb_norm_20x416x416x3.npy ]; then
    generate_npy $VBX_SDK/tutorials/coco2017_rgb_20x416x416x3.npy -o $VBX_SDK/tutorials/coco2017_rgb_norm_20x416x416x3.npy -s 416 416  --norm 
fi

echo "Checking for yolov2-tiny files..."

# model details @ https://pjreddie.com/darknet/yolo/
[ -f coco.names ] || wget -q https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names
if [ ! -f yolov2-tiny.json ]; then
   [ -f yolov2-tiny.cfg ] || wget -q https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov2-tiny.cfg
   [ -f yolov2-tiny.weights ] || wget -q http://web.archive.org/web/20220325031036/https://pjreddie.com/media/files/yolov2-tiny.weights
python $VBX_SDK/tutorials/darknet/darknet_to_onnx.py yolov2-tiny.cfg
fi
    if [ ! -f calibration_image_sample_data_20x128x128x3_float32.npy ]; then
        wget -q --no-check-certificate https://github.com/Microchip-Vectorblox/assets/raw/refs/heads/main/npy_files/calibration_image_sample_data_20x128x128x3_float32.npy
    fi


if [ -f yolov2-tiny.cfg ]; then
   if ! echo "d0b61c65d80a1aa8f3a72e72a241d38c yolov2-tiny.cfg" | md5sum -c; then
       echo -e "\n There is an issue with the yolov2-tiny model file as the expected checksum does not match.\n The model source can be found at: https://pjreddie.com/darknet/yolo/.\n If the model information has changed, please update this script and re-run the tutorial."
       exit 1
   fi
fi


# onnx2tf is an external model conversion tool to convert an onnx model to int8 tflite
# specific operation information can be found here: https://pypi.org/project/onnx2tf/
#  Purpose: Convert source model to int8 tflite format
#  - Required Inputs: onnx compliant model, calibration npy array
#  - Output: int8 tflite model
if [ ! -f yolov2-tiny.tflite ]; then
   echo "Running ONNX2TF..."
   onnx2tf -cind X0 $VBX_SDK/tutorials/coco2017_rgb_norm_20x416x416x3.npy [[[[0.,0.,0.]]]] [[[[1.,1.,1.]]]] \
-b 1 \
-i yolov2-tiny.onnx \
--output_signaturedefs \
--output_integer_quantized_tflite
   cp saved_model/yolov2-tiny_full_integer_quant.tflite yolov2-tiny.tflite
fi

# tflite_preprocess is an internal tool used to add a preprocess layer to the start of the model
#  Purpose: adds a preprocess layer to the start of the model (if none, will just preprocess by adding a uint8->int8 layer)
#  - Required Inputs: tflite source model, additional arguments 
#  - Outputs: preprocessed tflite model
if [ -f yolov2-tiny.tflite ]; then
   tflite_preprocess yolov2-tiny.tflite  --scale 255
fi


# vnnx_compile is an internal tool that converts an int8 tflite file to a binary file that can be run on the SDK and VectorBlox FPGA
#  Purpose: converts int8 tflite to binary
#  - Required Inputs: int8 tflite, size configuration, compression configuration, output file name
#  - Outputs: binary object files(.hex and binary file)
if [ -f yolov2-tiny.pre.tflite ]; then
    echo "Generating VNNX for V1000 ncomp configuration..."
    vnnx_compile -s V1000 -c ncomp -t yolov2-tiny.pre.tflite  -o yolov2-tiny_V1000_ncomp.vnnx
fi


# This step runs the final compiled binary in Python, it also shows how to run the same file in C simulation for SDK
#   *Currently C simulation is not supported for unstructured compression
if [ -f yolov2-tiny_V1000_ncomp.vnnx ]; then
    echo "Running Simulation..."
    python $VBX_SDK/example/python/yoloInfer.py yolov2-tiny_V1000_ncomp.vnnx $VBX_SDK/tutorials/test_images/dog.jpg -j yolov2-tiny.json -v 2 -l coco.names 
    echo "C Simulation Command:"
    echo '$VBX_SDK/example/sim-c/sim-run-model yolov2-tiny_V1000_ncomp.vnnx $VBX_SDK/tutorials/test_images/dog.jpg YOLOV2'
fi

deactivate
