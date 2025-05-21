
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

echo "Downloading yolov8n-pose_512x288_split..."
# model details @ https://github.com/ultralytics/ultralytics/
if [ ! -f yolov8n-pose.onnx ]; then
   # ignore ultralytics yolo command error, we only care about the Tflite which is generated
   yolo export model=yolov8n-pose.pt format=onnx imgsz=288,512 || true
fi
python pose_split_kps.py
onnx2tf -cind images $VBX_SDK/tutorials/coco2017_rgb_20x288x512x3.npy [[[[128,128,128]]]] [[[[128,128,128]]]] -i yolov8n-pose_edit.onnx --output_signaturedefs --output_integer_quantized_tflite
cp saved_model/yolov8n-pose_edit_full_integer_quant.tflite yolov8n-pose_512x288_split.tflite


if [ -f yolov8n-pose_512x288_split.tflite ]; then
   tflite_preprocess yolov8n-pose_512x288_split.tflite  --scale 255
fi

if [ -f yolov8n-pose_512x288_split.pre.tflite ]; then
    echo "Generating VNNX for V1000 configuration..."
    vnnx_compile -c V1000 -t yolov8n-pose_512x288_split.pre.tflite -o yolov8n-pose_512x288_split.vnnx
fi

if [ -f yolov8n-pose_512x288_split.vnnx ]; then
    echo "Running Simulation..."
    python $VBX_SDK/example/python/ultralytics.py yolov8n-pose_512x288_split.vnnx $VBX_SDK/tutorials/test_images/ski.273.481.jpg --task pose -nc 1  
    echo "C Simulation Command:"
    echo '$VBX_SDK/example/sim-c/sim-run-model yolov8n-pose_512x288_split.vnnx $VBX_SDK/tutorials/test_images/ski.273.481.jpg ULTRALYTICS_POSE'
fi

deactivate
