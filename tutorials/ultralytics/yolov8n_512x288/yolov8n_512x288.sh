
##########################################################
#  _    __          __             ____  __              #
# | |  / /__  _____/ /_____  _____/ __ )/ /___  _  __    #
# | | / / _ \/ ___/ __/ __ \/ ___/ __  / / __ \| |/_/    #
# | |/ /  __/ /__/ /_/ /_/ / /  / /_/ / / /_/ />  <      #
# |___/\___/\___/\__/\____/_/  /_____/_/\____/_/|_|      #
#                                                        #
# https://github.com/Microchip-Vectorblox/VectorBlox-SDK #
# v3.0                                                   #
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

echo "Checking for yolov8n_512x288 files..."

# model details @ https://github.com/ultralytics/ultralytics/
[ -f coco.names ] || wget -q https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names
if [ ! -f yolov8n_512x288.tflite ]; then
   if [ ! -f yolov8n.pb ]; then
       # ignore ultralytics yolo command error, we only care about the Tflite which is generated
       yolo export model=yolov8n.pt format=pb imgsz=288,512 || true
   fi
   tflite_quantize yolov8n.pb yolov8n_512x288.tflite -d $VBX_SDK/tutorials/coco2017_rgb_20x288x512x3.npy --mean 128 --scale 128 --shape 1 288 512 3
fi



if [ -f yolov8n_512x288.tflite ]; then 
   echo "Cutting graph" 
   tflite_cut yolov8n_512x288.tflite -c 190 197 207 214 224 231
   mv yolov8n_512x288.0.tflite yolov8n_512x288.cut.tflite 
fi

if [ -f yolov8n_512x288.cut.tflite ]; then
   tflite_preprocess yolov8n_512x288.cut.tflite  --scale 255
fi

if [ -f yolov8n_512x288.cut.pre.tflite ]; then
    echo "Generating VNNX for V1000 ncomp configuration..."
    vnnx_compile -s V1000 -c ncomp -t yolov8n_512x288.cut.pre.tflite  -o yolov8n_512x288_V1000_ncomp.vnnx
fi

if [ -f yolov8n_512x288_V1000_ncomp.vnnx ]; then
    echo "Running Simulation..."
    python $VBX_SDK/example/python/yoloInfer.py yolov8n_512x288_V1000_ncomp.vnnx $VBX_SDK/tutorials/test_images/dog.512.288.jpg -v 8 -l coco.names 
    echo "C Simulation Command:"
    echo '$VBX_SDK/example/sim-c/sim-run-model yolov8n_512x288_V1000_ncomp.vnnx $VBX_SDK/tutorials/test_images/dog.512.288.jpg ULTRALYTICS'
fi

deactivate
