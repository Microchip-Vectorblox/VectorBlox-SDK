
##########################################################
#  _    __          __             ____  __              #
# | |  / /__  _____/ /_____  _____/ __ )/ /___  _  __    #
# | | / / _ \/ ___/ __/ __ \/ ___/ __  / / __ \| |/_/    #
# | |/ /  __/ /__/ /_/ /_/ / /  / /_/ / / /_/ />  <      #
# |___/\___/\___/\__/\____/_/  /_____/_/\____/_/|_|      #
#                                                        #
# https://github.com/Microchip-Vectorblox/VectorBlox-SDK #
# v2.0                                                   #
#                                                        #
##########################################################

set -e
echo "Checking and activating VBX Python Environment..."
if [ -z $VBX_SDK ]; then
    echo "\$VBX_SDK not set. Please run 'source setup_vars.sh' from the SDK's root folder" && exit 1
fi
source $VBX_SDK/vbx_env/bin/activate

echo "Downloading yolov8n-obb..."
# model details @ https://github.com/ultralytics/ultralytics/
[ -f coco.names ] || wget -q https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names
if [ ! -f yolov8n-obb.tflite ]; then
    # ignore ultralytics yolo command error, we only care about the Tflite which is generated
    yolo export model=yolov8n-obb.pt format=tflite int8 || true
fi
cp yolov8n-obb_saved_model/yolov8n-obb_full_integer_quant.tflite yolov8n-obb.tflite

tflite_cut yolov8n-obb.tflite -c 256 263 222 229 197 204 189 214 239
mv yolov8n-obb.0.tflite yolov8n-obb.tflite

if [ -f yolov8n-obb.tflite ]; then
   tflite_preprocess yolov8n-obb.tflite  --scale 255
fi

if [ -f yolov8n-obb.pre.tflite ]; then
    echo "Generating VNNX for V1000 configuration..."
    vnnx_compile -c V1000 -t yolov8n-obb.pre.tflite -o yolov8n-obb.vnnx
fi

if [ -f yolov8n-obb.vnnx ]; then
    echo "Running Simulation..."
    python $VBX_SDK/example/python/ultralytics.py yolov8n-obb.vnnx $VBX_SDK/tutorials/test_images/harbour.jpg --task obb -nc 15 
    echo "C Simulation Command:"
    echo '$VBX_SDK/example/sim-c/sim-run-model yolov8n-obb.vnnx $VBX_SDK/tutorials/test_images/harbour.jpg  '
fi

deactivate
