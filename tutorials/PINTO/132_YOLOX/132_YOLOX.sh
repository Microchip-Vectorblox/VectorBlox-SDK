
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

echo "Checking for 132_YOLOX files..."

# model details @ https://github.com/PINTO0309/PINTO_model_zoo
[ -f coco.names ] || wget -q https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names
[ -f Yolo-X.json ] || wget -q https://github.com/Microchip-Vectorblox/assets/releases/download/assets/Yolo-X.json
if ! [ -f 132_YOLOX.tflite ]; then
    mkdir -p temp
    cd temp
    wget -q https://raw.githubusercontent.com/PINTO0309/PINTO_model_zoo/refs/heads/main/132_YOLOX/download_tiny.sh
    bash download_tiny.sh
    cd ..
    cp temp/resources_new/saved_model_yolox_tiny_416x416/model_weight_quant.tflite 132_YOLOX.tflite
    rm -rf temp
fi


if [ -f 132_YOLOX.tflite ]; then 
   echo "Cutting graph" 
   tflite_cut 132_YOLOX.tflite -c 321 322 337 390 391 406 252 253 268
   mv 132_YOLOX.0.tflite 132_YOLOX.cut.tflite 
fi

if [ -f 132_YOLOX.cut.tflite ]; then
   tflite_preprocess 132_YOLOX.cut.tflite  --scale 255
fi

if [ -f 132_YOLOX.cut.pre.tflite ]; then
    echo "Generating VNNX for V1000 ncomp configuration..."
    vnnx_compile -s V1000 -c ncomp -t 132_YOLOX.cut.pre.tflite  -o 132_YOLOX_V1000_ncomp.vnnx
fi

if [ -f 132_YOLOX_V1000_ncomp.vnnx ]; then
    echo "Running Simulation..."
    python $VBX_SDK/example/python/yoloInfer.py 132_YOLOX_V1000_ncomp.vnnx $VBX_SDK/tutorials/test_images/dog.jpg -v x -j Yolo-X.json -l coco.names -t 0.3 
    echo "C Simulation Command:"
    echo '$VBX_SDK/example/sim-c/sim-run-model 132_YOLOX_V1000_ncomp.vnnx $VBX_SDK/tutorials/test_images/dog.jpg YOLOX'
fi

deactivate
