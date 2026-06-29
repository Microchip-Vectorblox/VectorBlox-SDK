
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


# Ultralytics YOLO models are available under the AGPL-3.0 open-source license.
# Projects that are not open source require an Ultralytics Enterprise License. To
# obtain a commercial license for R&D and production use without open-source obligations,
# please complete the licensing form at https://www.ultralytics.com/license.
    

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

echo "Checking for yolov8n-pose_512x288 files..."

# model details @ https://github.com/ultralytics/ultralytics/
if [ ! -f calibration_image_sample_data_20x128x128x3_float32.npy ]; then
   wget -q --no-check-certificate https://github.com/Microchip-Vectorblox/assets/raw/refs/heads/main/npy_files/calibration_image_sample_data_20x128x128x3_float32.npy
fi

if [ ! -f yolov8n-pose_512x288.tflite ]; then
   if [ ! -f yolov8n-pose.pb ]; then
       # ignore ultralytics yolo command error, we only care about the Tflite which is generated
       yolo export model=yolov8n-pose.pt format=pb imgsz=288,512 || true
   fi
   tflite_quantize yolov8n-pose.pb yolov8n-pose_512x288.tflite -d $VBX_SDK/tutorials/coco2017_rgb_20x288x512x3.npy --mean 128 --scale 128 --shape 1 288 512 3
fi



if [ -f yolov8n-pose.pt ]; then
   if ! echo "fce9c3a495cc42f597c8191798b1445b yolov8n-pose.pt" | md5sum -c; then
       echo -e "\n There is an issue with the yolov8n-pose_512x288 model file as the expected checksum does not match.\n The model source can be found at: https://github.com/ultralytics/ultralytics/.\n If the model information has changed, please update this script and re-run the tutorial."
       exit 1
   fi
fi


# tflite_cut is an internal tool used to split an existing model into smaller models
#  Purpose: cuts a model into smaller subsections, can be used to decrease runtime or for debugging purposes
#  - Required Inputs: tflite source model, cut section(s)
#  - Outputs: tflite model
if [ -f yolov8n-pose_512x288.tflite ]; then 
   echo "Cutting graph" 
   tflite_cut yolov8n-pose_512x288.tflite -c 189 215 241 198 205 224 231 261 268
   mv yolov8n-pose_512x288.0.tflite yolov8n-pose_512x288.cut.tflite 
fi


# tflite_preprocess is an internal tool used to add a preprocess layer to the start of the model
#  Purpose: adds a preprocess layer to the start of the model (if none, will just preprocess by adding a uint8->int8 layer)
#  - Required Inputs: tflite source model, additional arguments 
#  - Outputs: preprocessed tflite model
if [ -f yolov8n-pose_512x288.cut.tflite ]; then
   tflite_preprocess yolov8n-pose_512x288.cut.tflite  --scale 255
fi


# vnnx_compile is an internal tool that converts an int8 tflite file to a binary file that can be run on the SDK and VectorBlox FPGA
#  Purpose: converts int8 tflite to binary
#  - Required Inputs: int8 tflite, size configuration, compression configuration, output file name
#  - Outputs: binary object files(.hex and binary file)
if [ -f yolov8n-pose_512x288.cut.pre.tflite ]; then
    echo "Generating VNNX for V1000 ncomp configuration..."
    vnnx_compile -s V1000 -c ncomp -t yolov8n-pose_512x288.cut.pre.tflite  -o yolov8n-pose_512x288_V1000_ncomp.vnnx
fi


# This step runs the final compiled binary in Python, it also shows how to run the same file in C simulation for SDK
#   *Currently C simulation is not supported for unstructured compression
if [ -f yolov8n-pose_512x288_V1000_ncomp.vnnx ]; then
    echo "Running Simulation..."
    python $VBX_SDK/example/python/ultralyticsInfer.py yolov8n-pose_512x288_V1000_ncomp.vnnx $VBX_SDK/tutorials/test_images/ski.273.481.jpg --task pose -nc 1  
    echo "C Simulation Command:"
    echo '$VBX_SDK/example/sim-c/sim-run-model yolov8n-pose_512x288_V1000_ncomp.vnnx $VBX_SDK/tutorials/test_images/ski.273.481.jpg POSE_DETECT'
fi

deactivate
