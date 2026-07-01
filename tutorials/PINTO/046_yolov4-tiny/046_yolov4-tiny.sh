
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

echo "Checking for 046_yolov4-tiny files..."

# model details @ https://github.com/PINTO0309/PINTO_model_zoo
[ -f voc.names ] || wget -q https://raw.githubusercontent.com/pjreddie/darknet/master/data/voc.names
[ -f yolo-v4-tiny-voc.json ] || wget -q https://github.com/Microchip-Vectorblox/assets/releases/download/assets/yolo-v4-tiny-voc.json
if ! [ -f 046_yolov4-tiny.tflite ]; then
    mkdir -p temp
    cd temp
    wget -q https://raw.githubusercontent.com/PINTO0309/PINTO_model_zoo/refs/heads/main/046_yolov4-tiny/download.sh
    bash download.sh
    cd ..
    pushd temp/05_full_integer_quantization; tar -zxf resources.tar.gz; popd
    cp temp/05_full_integer_quantization/yolov4_tiny_voc_416x416_full_integer_quant.tflite 046_yolov4-tiny.tflite
    rm -rf temp
fi


if [ -f 046_yolov4-tiny.tflite ]; then
   if ! echo "9178bf4c75fd24e287c816c5af71cff1 046_yolov4-tiny.tflite" | md5sum -c; then
       echo -e "\n There is an issue with the 046_yolov4-tiny model file as the expected checksum does not match.\n The model source can be found at: https://github.com/PINTO0309/PINTO_model_zoo.\n If the model information has changed, please update this script and re-run the tutorial."
       exit 1
   fi
fi


# tflite_preprocess is an internal tool used to add a preprocess layer to the start of the model
#  Purpose: adds a preprocess layer to the start of the model (if none, will just preprocess by adding a uint8->int8 layer)
#  - Required Inputs: tflite source model, additional arguments 
#  - Outputs: preprocessed tflite model
if [ -f 046_yolov4-tiny.tflite ]; then
   tflite_preprocess 046_yolov4-tiny.tflite  --scale 255
fi


# vnnx_compile is an internal tool that converts an int8 tflite file to a binary file that can be run on the SDK and VectorBlox FPGA
#  Purpose: converts int8 tflite to binary
#  - Required Inputs: int8 tflite, size configuration, compression configuration, output file name
#  - Outputs: binary object files(.hex and binary file)
if [ -f 046_yolov4-tiny.pre.tflite ]; then
    echo "Generating VNNX for V1000 ncomp configuration..."
    vnnx_compile -s V1000 -c ncomp -t 046_yolov4-tiny.pre.tflite  -o 046_yolov4-tiny_V1000_ncomp.vnnx
fi


# This step runs the final compiled binary in Python, it also shows how to run the same file in C simulation for SDK
#   *Currently C simulation is not supported for unstructured compression
if [ -f 046_yolov4-tiny_V1000_ncomp.vnnx ]; then
    echo "Running Simulation..."
    python $VBX_SDK/example/python/yoloInfer.py 046_yolov4-tiny_V1000_ncomp.vnnx $VBX_SDK/tutorials/test_images/dog.jpg -j yolo-v4-tiny-voc.json -v 3 -l voc.names -i 0.3 
    echo "C Simulation Command:"
    echo '$VBX_SDK/example/sim-c/sim-run-model 046_yolov4-tiny_V1000_ncomp.vnnx $VBX_SDK/tutorials/test_images/dog.jpg '
fi

deactivate
