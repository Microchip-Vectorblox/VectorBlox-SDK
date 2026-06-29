
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
if [ ! -f $VBX_SDK/tutorials/coco2017_rgb_norm_20x640x640x3.npy ]; then
    generate_npy $VBX_SDK/tutorials/coco2017_rgb_20x416x416x3.npy -o $VBX_SDK/tutorials/coco2017_rgb_norm_20x640x640x3.npy -s 640 640  --norm 
fi

echo "Checking for yolov8n_comp66 files..."

# model details @ https://github.com/ultralytics/ultralytics/
[ -f coco.names ] || wget -q https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names
if [ ! -f yolov8n_comp66.tflite ]; then
   wget -q https://github.com/Microchip-Vectorblox/assets/releases/download/assets/yolov8n_comp66.onnx
fi
    if [ ! -f calibration_image_sample_data_20x128x128x3_float32.npy ]; then
        wget -q --no-check-certificate https://github.com/Microchip-Vectorblox/assets/raw/refs/heads/main/npy_files/calibration_image_sample_data_20x128x128x3_float32.npy
    fi


if [ -f yolov8n_comp66.onnx ]; then
   if ! echo "9dbeb3ee69033dda1dd765e221b75fc7 yolov8n_comp66.onnx" | md5sum -c; then
       echo -e "\n There is an issue with the yolov8n_comp66 model file as the expected checksum does not match.\n The model source can be found at: https://github.com/ultralytics/ultralytics/.\n If the model information has changed, please update this script and re-run the tutorial."
       exit 1
   fi
fi


# onnx2tf is an external model conversion tool to convert an onnx model to int8 tflite
# specific operation information can be found here: https://pypi.org/project/onnx2tf/
#  Purpose: Convert source model to int8 tflite format
#  - Required Inputs: onnx compliant model, calibration npy array
#  - Output: int8 tflite model
if [ ! -f yolov8n_comp66.tflite ]; then
   echo "Running ONNX2TF..."
   onnx2tf -cind images $VBX_SDK/tutorials/coco2017_rgb_norm_20x640x640x3.npy [[[[0.,0.,0.]]]] [[[[1.,1.,1.]]]] \
--disable_group_convolution \
--output_names_to_interrupt_model_conversion "/model.22/cv2.0/cv2.0.2/Conv_output_0" "/model.22/cv3.0/cv3.0.2/Conv_output_0" "/model.22/cv2.1/cv2.1.2/Conv_output_0" "/model.22/cv3.1/cv3.1.2/Conv_output_0" "/model.22/cv2.2/cv2.2.2/Conv_output_0" "/model.22/cv3.2/cv3.2.2/Conv_output_0" \
-i yolov8n_comp66.onnx \
--output_signaturedefs \
--output_integer_quantized_tflite
   cp saved_model/yolov8n_comp66_full_integer_quant.tflite yolov8n_comp66.tflite
fi

# tflite_preprocess is an internal tool used to add a preprocess layer to the start of the model
#  Purpose: adds a preprocess layer to the start of the model (if none, will just preprocess by adding a uint8->int8 layer)
#  - Required Inputs: tflite source model, additional arguments 
#  - Outputs: preprocessed tflite model
if [ -f yolov8n_comp66.tflite ]; then
   tflite_preprocess yolov8n_comp66.tflite  --scale 255
fi


# vnnx_compile is an internal tool that converts an int8 tflite file to a binary file that can be run on the SDK and VectorBlox FPGA
#  Purpose: converts int8 tflite to binary
#  - Required Inputs: int8 tflite, size configuration, compression configuration, output file name
#  - Outputs: binary object files(.hex and binary file)
if [ -f yolov8n_comp66.pre.tflite ]; then
    echo "Generating VNNX for V1000 comp configuration..."
    vnnx_compile -s V1000 -c comp -t yolov8n_comp66.pre.tflite  -o yolov8n_comp66_V1000_comp.vnnx
fi


# This step runs the final compiled binary in Python, it also shows how to run the same file in C simulation for SDK
#   *Currently C simulation is not supported for unstructured compression
if [ -f yolov8n_comp66_V1000_comp.vnnx ]; then
    echo "Running Simulation..."
    python $VBX_SDK/example/python/yoloInfer.py yolov8n_comp66_V1000_comp.vnnx $VBX_SDK/tutorials/test_images/dog.jpg -v 8 -l coco.names 
    echo "C Simulation Command:"
    echo '$VBX_SDK/example/sim-c/sim-run-model yolov8n_comp66_V1000_comp.vnnx $VBX_SDK/tutorials/test_images/dog.jpg OBJECT_DETECT'
fi

deactivate
