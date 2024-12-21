
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

echo "Downloading yolov8s_cut..."
# model details @ https://github.com/ultralytics/ultralytics/
[ -f coco.names ] || wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names
if [ ! -f yolov8s.tflite ]; then
    yolo export model=yolov8s.pt format=tflite int8 || true
fi
cp yolov8s_saved_model/yolov8s_full_integer_quant.tflite yolov8s_cut.tflite

tflite_cut yolov8s_cut.tflite -c 190 197 207 214 224 231
mv yolov8s_cut.0.tflite yolov8s_cut.tflite

if [ -f yolov8s_cut.tflite ]; then
   tflite_preprocess yolov8s_cut.tflite  --scale 255
fi

if [ -f yolov8s_cut.pre.tflite ]; then
    echo "Generating VNNX for V1000 configuration..."
    vnnx_compile -c V1000 -t yolov8s_cut.pre.tflite -o yolov8s_cut.vnnx
fi

if [ -f yolov8s_cut.vnnx ]; then
    echo "Running Simulation..."
    python $VBX_SDK/example/python/yoloInfer.py yolov8s_cut.vnnx $VBX_SDK/tutorials/test_images/dog.jpg -v 8 -l coco.names 
fi

deactivate
