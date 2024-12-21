
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

echo "Downloading yolov8n_cut..."
# model details @ https://github.com/ultralytics/ultralytics/
[ -f coco.names ] || wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names
if [ ! -f yolov8n.tflite ]; then
    # ignore ultralytics yolo command error, we only care about the Tflite which is generated
    yolo export model=yolov8n.pt format=tflite int8 || true
fi
cp yolov8n_saved_model/yolov8n_full_integer_quant.tflite yolov8n_cut.tflite

tflite_cut yolov8n_cut.tflite -c 189 196 206 213 223 230
mv yolov8n_cut.0.tflite yolov8n_cut.tflite

if [ -f yolov8n_cut.tflite ]; then
   tflite_preprocess yolov8n_cut.tflite  --scale 255
fi

if [ -f yolov8n_cut.pre.tflite ]; then
    echo "Generating VNNX for V1000 configuration..."
    vnnx_compile -c V1000 -t yolov8n_cut.pre.tflite -o yolov8n_cut.vnnx
fi

if [ -f yolov8n_cut.vnnx ]; then
    echo "Running Simulation..."
    python $VBX_SDK/example/python/yoloInfer.py yolov8n_cut.vnnx $VBX_SDK/tutorials/test_images/dog.jpg -v 8 -l coco.names 
fi

deactivate
