
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
if [ ! -f $VBX_SDK/tutorials/coco2017_rgb_20x288x512x3.npy ]; then
    generate_npy $VBX_SDK/tutorials/coco2017_rgb_20x416x416x3.npy -o $VBX_SDK/tutorials/coco2017_rgb_20x288x512x3.npy -s 288 512 
fi

echo "Checking for yolov8n_50s_25p_512x288 files..."

# model details @ https://github.com/ultralytics/ultralytics/
[ -f coco.names ] || wget -q https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names
if [ ! -f yolov8n_50s_25p_512x288.tflite ]; then
   wget -q https://github.com/Microchip-Vectorblox/assets/releases/download/assets/yolov8n_50s_25p_512x288.tflite
fi




# vnnx_compile is an internal tool that converts an int8 tflite file to a binary file that can be run on the SDK and VectorBlox FPGA
#  Purpose: converts int8 tflite to binary
#  - Required Inputs: int8 tflite, size configuration, compression configuration, output file name
#  - Outputs: binary object files(.hex and binary file)
if [ -f yolov8n_50s_25p_512x288.tflite ]; then
    echo "Generating VNNX for V1000 ucomp configuration..."
    vnnx_compile -s V1000 -c ucomp -t yolov8n_50s_25p_512x288.tflite --uint8 -o yolov8n_50s_25p_512x288.ucomp
fi


# This step runs the final compiled binary in Python, it also shows how to run the same file in C simulation for SDK
#   *Currently C simulation is not supported for unstructured compression
if [ -f yolov8n_50s_25p_512x288.ucomp ]; then
    echo "Running Simulation..."
    python $VBX_SDK/example/python/yoloInfer.py yolov8n_50s_25p_512x288.ucomp $VBX_SDK/tutorials/test_images/dog.512.288.jpg -v 8 -l coco.names 
    echo "C Simulation Command:"
    echo '$VBX_SDK/example/sim-c/sim-run-model yolov8n_50s_25p_512x288.ucomp $VBX_SDK/tutorials/test_images/dog.512.288.jpg OBJECT_DETECT'
fi

deactivate
