
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

echo "Checking for yolov5n_70s_512x512 files..."

# model details @ https://github.com/ultralytics/ultralytics/
[ -f coco.names ] || wget -q https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names
[ -f yolov5n_512x512.json ] || wget -q https://github.com/Microchip-Vectorblox/assets/releases/download/assets/yolov5n_512x512.json
if [ ! -f yolov5n_70s_512x512.tflite ]; then
   wget -q https://github.com/Microchip-Vectorblox/assets/releases/download/assets/yolov5n_70s_512x512.tflite
fi


if [ -f yolov5n_70s_512x512.tflite ]; then
    echo "Generating VNNX for V1000 ucomp configuration..."
    vnnx_compile -s V1000 -c ucomp -t yolov5n_70s_512x512.tflite --uint8 -o yolov5n_70s_512x512.ucomp
fi

if [ -f yolov5n_70s_512x512.ucomp ]; then
    echo "Running Simulation..."
    python $VBX_SDK/example/python/yoloInfer.py yolov5n_70s_512x512.ucomp $VBX_SDK/tutorials/test_images/dog.jpg -j yolov5n_512x512.json -v 5 -l coco.names -t 0.25 
    echo "C Simulation Command:"
    echo '$VBX_SDK/example/sim-c/sim-run-model yolov5n_70s_512x512.ucomp $VBX_SDK/tutorials/test_images/dog.jpg YOLOV5'
fi

deactivate
