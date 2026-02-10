
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

echo "Checking for yolov8n_pose_50s_25p_512x288_split files..."

# model details @ https://github.com/ultralytics/ultralytics/
[ -f coco.names ] || wget -q https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names
if [ ! -f yolov8n_pose_50s_25p_512x288_split.tflite ]; then
   wget -q https://github.com/Microchip-Vectorblox/assets/releases/download/assets/yolov8n_pose_50s_25p_512x288_split.tflite
fi



if [ -f yolov8n_pose_50s_25p_512x288_split.tflite ]; then
    echo "Generating VNNX for V1000 ucomp configuration..."
    vnnx_compile -s V1000 -c ucomp -t yolov8n_pose_50s_25p_512x288_split.tflite --uint8 -o yolov8n_pose_50s_25p_512x288_split.ucomp
fi

if [ -f yolov8n_pose_50s_25p_512x288_split.ucomp ]; then
    echo "Running Simulation..."
    python $VBX_SDK/example/python/ultralyticsInfer.py yolov8n_pose_50s_25p_512x288_split.ucomp $VBX_SDK/tutorials/test_images/ski.273.481.jpg --task pose -nc 1 
    echo "C Simulation Command:"
    echo '$VBX_SDK/example/sim-c/sim-run-model yolov8n_pose_50s_25p_512x288_split.ucomp $VBX_SDK/tutorials/test_images/ski.273.481.jpg ULTRALYTICS'
fi

deactivate
