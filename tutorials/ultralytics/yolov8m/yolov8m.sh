
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

echo "Downloading yolov8m..."
# model details @ https://github.com/ultralytics/ultralytics/
[ -f coco.names ] || wget -q https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names
if [ ! -f yolov8m.tflite ]; then
    # ignore ultralytics yolo command error, we only care about the Tflite which is generated
    yolo export model=yolov8m.pt format=tflite int8 || true
    cp yolov8m_saved_model/yolov8m_full_integer_quant.tflite yolov8m.tflite
fi

tflite_cut yolov8m.tflite -c 256 263 273 280 290 297
mv yolov8m.0.tflite yolov8m.tflite

if [ -f yolov8m.tflite ]; then
   tflite_preprocess yolov8m.tflite  --scale 255
fi

if [ -f yolov8m.pre.tflite ]; then
    echo "Generating VNNX for V1000 configuration..."
    vnnx_compile -c V1000 -t yolov8m.pre.tflite -o yolov8m.vnnx
fi

if [ -f yolov8m.vnnx ]; then
    echo "Running Simulation..."
    python $VBX_SDK/example/python/yoloInfer.py yolov8m.vnnx $VBX_SDK/tutorials/test_images/dog.jpg -v 8 -l coco.names 
    echo "C Simulation Command:"
    echo '$VBX_SDK/example/sim-c/sim-run-model yolov8m.vnnx $VBX_SDK/tutorials/test_images/dog.jpg ULTRALYTICS'
fi

deactivate
