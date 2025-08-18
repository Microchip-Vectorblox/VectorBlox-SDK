
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

echo "Checking for Yolo-X-Quantized files..."

# model details @ https://aihub.qualcomm.com/models/YOLO
[ -f coco.names ] || wget -q https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names
if [ ! -f Yolo-X-Quantized.tflite ]; then
   wget -q --no-check-certificate https://huggingface.co/qualcomm/Yolo-X/resolve/v0.32.0/Yolo-X_w8a8.tflite
   mv Yolo-X_w8a8.tflite Yolo-X-Quantized.tflite
fi


if [ -f Yolo-X-Quantized.tflite ]; then 
   echo "Cutting graph" 
   tflite_cut Yolo-X-Quantized.tflite -c 193 239 285 195 184 241 230 314 276
   mv Yolo-X-Quantized.0.tflite Yolo-X-Quantized.cut.tflite 
fi

if [ -f Yolo-X-Quantized.cut.tflite ]; then
   tflite_preprocess Yolo-X-Quantized.cut.tflite  --scale 255
fi

if [ -f Yolo-X-Quantized.cut.pre.tflite ]; then
    echo "Generating VNNX for V1000 configuration..."
    vnnx_compile -c V1000 -t Yolo-X-Quantized.cut.pre.tflite -o Yolo-X-Quantized.vnnx
fi

if [ -f Yolo-X-Quantized.vnnx ]; then
    echo "Running Simulation..."
    python $VBX_SDK/example/python/yoloInfer.py Yolo-X-Quantized.vnnx $VBX_SDK/tutorials/test_images/dog.jpg -v x -j Yolo-X.json -l coco.names 
    echo "C Simulation Command:"
    echo '$VBX_SDK/example/sim-c/sim-run-model Yolo-X-Quantized.vnnx $VBX_SDK/tutorials/test_images/dog.jpg YOLOX'
fi

deactivate
