
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

echo "Downloading yolov3-tinyu..."
# model details @ https://github.com/ultralytics/ultralytics/
[ -f coco.names ] || wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names
if [ ! -f yolov3-tinyu.tflite ]; then
    # ignore ultralytics yolo command error, we only care about the Tflite which is generated
    yolo export model=yolov3-tinyu.pt format=tflite int8 || true
    cp yolov3-tinyu_saved_model/yolov3-tinyu_full_integer_quant.tflite yolov3-tinyu.tflite
fi

if [ -f yolov3-tinyu.tflite ]; then
   tflite_preprocess yolov3-tinyu.tflite  --scale 255
fi

if [ -f yolov3-tinyu.pre.tflite ]; then
    echo "Generating VNNX for V1000 configuration..."
    vnnx_compile -c V1000 -t yolov3-tinyu.pre.tflite -o yolov3-tinyu.vnnx
fi

if [ -f yolov3-tinyu.vnnx ]; then
    echo "Running Simulation..."
    python $VBX_SDK/example/python/yoloInfer.py yolov3-tinyu.vnnx $VBX_SDK/tutorials/test_images/dog.jpg -v 8 -l coco.names 
fi

deactivate
