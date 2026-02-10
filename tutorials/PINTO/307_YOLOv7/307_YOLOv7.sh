
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

echo "Checking for 307_YOLOv7 files..."

# model details @ https://github.com/PINTO0309/PINTO_model_zoo
[ -f coco.names ] || wget -q https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names
[ -f yolov5n.json ] || wget -q https://github.com/Microchip-Vectorblox/assets/releases/download/assets/yolov5n.json
if ! [ -f 307_YOLOv7.tflite ]; then
    mkdir -p temp
    cd temp
    wget -q https://raw.githubusercontent.com/PINTO0309/PINTO_model_zoo/refs/heads/main/307_YOLOv7/download_openvino_myriad_oakd_tf_tflite.sh
    bash download_openvino_myriad_oakd_tf_tflite.sh
    cd ..
    cp temp/yolov7-tiny_384x640/model_integer_quant.tflite 307_YOLOv7.tflite
    rm -rf temp
fi


if [ -f 307_YOLOv7.tflite ]; then 
   echo "Cutting graph" 
   tflite_cut 307_YOLOv7.tflite -c 0 119 155 191
   mv 307_YOLOv7.1.tflite 307_YOLOv7.cut.tflite 
fi

if [ -f 307_YOLOv7.cut.tflite ]; then
   tflite_preprocess 307_YOLOv7.cut.tflite  --scale 255
fi

if [ -f 307_YOLOv7.cut.pre.tflite ]; then
    echo "Generating VNNX for V1000 ncomp configuration..."
    vnnx_compile -s V1000 -c ncomp -t 307_YOLOv7.cut.pre.tflite  -o 307_YOLOv7_V1000_ncomp.vnnx
fi

if [ -f 307_YOLOv7_V1000_ncomp.vnnx ]; then
    echo "Running Simulation..."
    python $VBX_SDK/example/python/yoloInfer.py 307_YOLOv7_V1000_ncomp.vnnx $VBX_SDK/tutorials/test_images/dog.jpg -v 7 -t 0.3 -j yolov5n.json -l coco.names 
    echo "C Simulation Command:"
    echo '$VBX_SDK/example/sim-c/sim-run-model 307_YOLOv7_V1000_ncomp.vnnx $VBX_SDK/tutorials/test_images/dog.jpg YOLOV5'
fi

deactivate
