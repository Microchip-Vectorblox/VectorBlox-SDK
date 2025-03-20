
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

echo "Downloading yolov9t..."
# model details @ https://github.com/ultralytics/ultralytics/
[ -f coco.names ] || wget -q https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names
if [ ! -f yolov9t.tflite ]; then
    # ignore ultralytics yolo command error, we only care about the Tflite which is generated
    yolo export model=yolov9t.pt format=tflite int8 || true
fi
cp yolov9t_saved_model/yolov9t_full_integer_quant.tflite yolov9t.tflite

tflite_cut yolov9t.tflite -c 611 618 628 635 645 652
mv yolov9t.0.tflite yolov9t.tflite

if [ -f yolov9t.tflite ]; then
   tflite_preprocess yolov9t.tflite  --scale 255
fi

if [ -f yolov9t.pre.tflite ]; then
    echo "Generating VNNX for V1000 configuration..."
    vnnx_compile -c V1000 -t yolov9t.pre.tflite -o yolov9t.vnnx
fi

if [ -f yolov9t.vnnx ]; then
    echo "Running Simulation..."
    python $VBX_SDK/example/python/yoloInfer.py yolov9t.vnnx $VBX_SDK/tutorials/test_images/dog.jpg -v 8 -l coco.names 
    echo "C Simulation Command:"
    echo '$VBX_SDK/example/sim-c/sim-run-model yolov9t.vnnx $VBX_SDK/tutorials/test_images/dog.jpg ULTRALYTICS'
fi

deactivate
