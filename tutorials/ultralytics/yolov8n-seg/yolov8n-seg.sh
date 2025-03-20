
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

echo "Downloading yolov8n-seg..."
# model details @ https://github.com/ultralytics/ultralytics/
[ -f coco.names ] || wget -q https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names
if [ ! -f yolov8n-seg.tflite ]; then
    # ignore ultralytics yolo command error, we only care about the Tflite which is generated
    yolo export model=yolov8n-seg.pt format=tflite int8 || true
fi
cp yolov8n-seg_saved_model/yolov8n-seg_full_integer_quant.tflite yolov8n-seg.tflite

tflite_cut yolov8n-seg.tflite -c 205 198 251 215 189 231 224 268 261 244
mv yolov8n-seg.0.tflite yolov8n-seg.tflite

if [ -f yolov8n-seg.tflite ]; then
   tflite_preprocess yolov8n-seg.tflite  --scale 255
fi

if [ -f yolov8n-seg.pre.tflite ]; then
    echo "Generating VNNX for V1000 configuration..."
    vnnx_compile -c V1000 -t yolov8n-seg.pre.tflite -o yolov8n-seg.vnnx
fi

if [ -f yolov8n-seg.vnnx ]; then
    echo "Running Simulation..."
    python $VBX_SDK/example/python/ultralytics.py yolov8n-seg.vnnx $VBX_SDK/tutorials/test_images/dog.jpg -l coco.names --task seg 
    echo "C Simulation Command:"
    echo '$VBX_SDK/example/sim-c/sim-run-model yolov8n-seg.vnnx $VBX_SDK/tutorials/test_images/dog.jpg  '
fi

deactivate
