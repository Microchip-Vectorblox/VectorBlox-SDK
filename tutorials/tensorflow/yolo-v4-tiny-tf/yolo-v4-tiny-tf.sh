
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
if [ ! -f $VBX_SDK/tutorials/coco2017_rgb_20x416x416x3.npy ]; then
    generate_npy $VBX_SDK/tutorials/coco2017_rgb_20x416x416x3.npy -o $VBX_SDK/tutorials/coco2017_rgb_20x416x416x3.npy -s 416 416 
fi

echo "Checking for yolo-v4-tiny-tf files..."

# model details @ https://github.com/openvinotoolkit/open_model_zoo/tree/2021.4.2/models/public/yolo-v4-tiny-tf/
[ -f coco.names ] || wget -q https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names
if [ ! -f yolo-v4-tiny-tf.tflite ]; then
   omz_downloader --name yolo-v4-tiny-tf
   DOWNLOAD_DIR=public/yolo-v4-tiny-tf
   python $DOWNLOAD_DIR/keras-YOLOv3-model-set/tools/model_converter/convert.py $DOWNLOAD_DIR/keras-YOLOv3-model-set/cfg/yolov4-tiny.cfg $DOWNLOAD_DIR/yolov4-tiny.weights yolo-v4-tiny.h5
fi



if [ -f yolo-v4-tiny.h5 ]; then
   if ! echo "8911bf808aed305d6854c4ea48fcc731 public/yolo-v4-tiny-tf/yolov4-tiny.weights" | md5sum -c; then
       echo -e "\n There is an issue with the yolo-v4-tiny-tf model file as the expected checksum does not match.\n The model source can be found at: https://github.com/openvinotoolkit/open_model_zoo/tree/2021.4.2/models/public/yolo-v4-tiny-tf/.\n If the model information has changed, please update this script and re-run the tutorial."
       exit 1
   fi
fi


# tflite_quantize is an internal tool designed to quantize a saved_model directory using the npy array
#  Purpose: Convert source model directory to int8 tflite format
#  - Required Inputs: model source directory, calibration npy array
#  - Output: int8 tflite model
if [ ! -f yolo-v4-tiny-tf.tflite ]; then
   echo "Generating TF Lite..."
   tflite_quantize yolo-v4-tiny.h5 yolo-v4-tiny-tf.tflite -d $VBX_SDK/tutorials/coco2017_rgb_20x416x416x3.npy \
--scale 255. --shape 1 416 416 3
fi


# tflite_preprocess is an internal tool used to add a preprocess layer to the start of the model
#  Purpose: adds a preprocess layer to the start of the model (if none, will just preprocess by adding a uint8->int8 layer)
#  - Required Inputs: tflite source model, additional arguments 
#  - Outputs: preprocessed tflite model
if [ -f yolo-v4-tiny-tf.tflite ]; then
   tflite_preprocess yolo-v4-tiny-tf.tflite  --scale 255
fi


# vnnx_compile is an internal tool that converts an int8 tflite file to a binary file that can be run on the SDK and VectorBlox FPGA
#  Purpose: converts int8 tflite to binary
#  - Required Inputs: int8 tflite, size configuration, compression configuration, output file name
#  - Outputs: binary object files(.hex and binary file)
if [ -f yolo-v4-tiny-tf.pre.tflite ]; then
    echo "Generating VNNX for V1000 ncomp configuration..."
    vnnx_compile -s V1000 -c ncomp -t yolo-v4-tiny-tf.pre.tflite  -o yolo-v4-tiny-tf_V1000_ncomp.vnnx
fi


# This step runs the final compiled binary in Python, it also shows how to run the same file in C simulation for SDK
#   *Currently C simulation is not supported for unstructured compression
if [ -f yolo-v4-tiny-tf_V1000_ncomp.vnnx ]; then
    echo "Running Simulation..."
    python $VBX_SDK/example/python/yoloInfer.py yolo-v4-tiny-tf_V1000_ncomp.vnnx $VBX_SDK/tutorials/test_images/dog.jpg -j yolo-v4-tiny-tf.json -v 3 -l coco.names -i 0.3 
    echo "C Simulation Command:"
    echo '$VBX_SDK/example/sim-c/sim-run-model yolo-v4-tiny-tf_V1000_ncomp.vnnx $VBX_SDK/tutorials/test_images/dog.jpg YOLOV4'
fi

deactivate
