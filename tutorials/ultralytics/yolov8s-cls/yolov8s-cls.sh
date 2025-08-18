
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

echo "Checking for yolov8s-cls files..."

# model details @ https://github.com/ultralytics/ultralytics/
if [ ! -f yolov8s-cls.tflite ]; then
    yolo export model=yolov8s-cls.pt format=tflite int8 || true
    cp yolov8s-cls_saved_model/yolov8s-cls_full_integer_quant.tflite yolov8s-cls.tflite
fi


if [ -f yolov8s-cls.tflite ]; then
   tflite_preprocess yolov8s-cls.tflite  --scale 255.
fi

if [ -f yolov8s-cls.pre.tflite ]; then
    echo "Generating VNNX for V1000 configuration..."
    vnnx_compile -c V1000 -t yolov8s-cls.pre.tflite -o yolov8s-cls.vnnx
fi

if [ -f yolov8s-cls.vnnx ]; then
    echo "Running Simulation..."
    python $VBX_SDK/example/python/classifier.py yolov8s-cls.vnnx $VBX_SDK/tutorials/test_images/oreo.jpg 
    echo "C Simulation Command:"
    echo '$VBX_SDK/example/sim-c/sim-run-model yolov8s-cls.vnnx $VBX_SDK/tutorials/test_images/oreo.jpg CLASSIFY'
fi

deactivate
