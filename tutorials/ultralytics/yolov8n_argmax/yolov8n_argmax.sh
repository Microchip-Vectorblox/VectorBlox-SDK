
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

echo "Checking for yolov8n_argmax files..."

# model details @ https://github.com/ultralytics/ultralytics/
[ -f coco.names ] || wget -q https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names
if [ ! -f yolov8n_argmax.tflite ]; then
    # ignore ultralytics yolo command error, we only care about the Tflite which is generated
    yolo export model=yolov8n.pt format=tflite int8 || true
    cp yolov8n_saved_model/yolov8n_full_integer_quant.tflite yolov8n_argmax.tflite
fi


if [ -f yolov8n_argmax.tflite ]; then 
   echo "Cutting graph" 
   tflite_cut yolov8n_argmax.tflite -c 189 196 206 213 223 230
   mv yolov8n_argmax.0.tflite yolov8n_argmax.cut.tflite 
fi

if [ -f yolov8n_argmax.cut.tflite ]; then
   tflite_preprocess yolov8n_argmax.cut.tflite  --scale 255
fi

if [ -f yolov8n_argmax.cut.pre.tflite ]; then
   tflite_postprocess yolov8n_argmax.cut.pre.tflite   -p YOLO_ARG_MAX
fi

if [ -f yolov8n_argmax.cut.pre.post.tflite ]; then
    echo "Generating VNNX for V1000 configuration..."
    vnnx_compile -c V1000 -t yolov8n_argmax.cut.pre.post.tflite -o yolov8n_argmax.vnnx
fi

if [ -f yolov8n_argmax.vnnx ]; then
    echo "Running Simulation..."
    python $VBX_SDK/example/python/yoloInfer.py yolov8n_argmax.vnnx $VBX_SDK/tutorials/test_images/dog.jpg -v 8 -l coco.names 
    echo "C Simulation Command:"
    echo '$VBX_SDK/example/sim-c/sim-run-model yolov8n_argmax.vnnx $VBX_SDK/tutorials/test_images/dog.jpg ULTRALYTICS'
fi

deactivate
