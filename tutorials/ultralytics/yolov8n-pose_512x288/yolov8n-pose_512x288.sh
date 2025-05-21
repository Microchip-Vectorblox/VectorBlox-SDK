
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

echo "Checking for Numpy calibration data file..."
if [ ! -f $VBX_SDK/tutorials/coco2017_rgb_20x288x512x3.npy ]; then
    generate_npy $VBX_SDK/tutorials/coco2017_rgb_20x416x416x3.npy -o $VBX_SDK/tutorials/coco2017_rgb_20x288x512x3.npy -s 288 512 
fi

echo "Downloading yolov8n-pose_512x288..."
# model details @ https://github.com/ultralytics/ultralytics/
if [ ! -f yolov8n-pose.pb ]; then
   # ignore ultralytics yolo command error, we only care about the Tflite which is generated
   yolo export model=yolov8n-pose.pt format=pb imgsz=288,512 || true
fi
tflite_quantize yolov8n-pose.pb yolov8n-pose_512x288.tflite -d $VBX_SDK/tutorials/coco2017_rgb_20x288x512x3.npy --mean 128 --scale 128 --shape 1 288 512 3


tflite_cut yolov8n-pose_512x288.tflite -c 189 215 241 198 205 224 231 261 268
mv yolov8n-pose_512x288.0.tflite yolov8n-pose_512x288.tflite

if [ -f yolov8n-pose_512x288.tflite ]; then
   tflite_preprocess yolov8n-pose_512x288.tflite  --scale 255
fi

if [ -f yolov8n-pose_512x288.pre.tflite ]; then
    echo "Generating VNNX for V1000 configuration..."
    vnnx_compile -c V1000 -t yolov8n-pose_512x288.pre.tflite -o yolov8n-pose_512x288.vnnx
fi

if [ -f yolov8n-pose_512x288.vnnx ]; then
    echo "Running Simulation..."
    python $VBX_SDK/example/python/ultralytics.py yolov8n-pose_512x288.vnnx $VBX_SDK/tutorials/test_images/ski.273.481.jpg --task pose -nc 1  
    echo "C Simulation Command:"
    echo '$VBX_SDK/example/sim-c/sim-run-model yolov8n-pose_512x288.vnnx $VBX_SDK/tutorials/test_images/ski.273.481.jpg ULTRALYTICS_POSE'
fi

deactivate
