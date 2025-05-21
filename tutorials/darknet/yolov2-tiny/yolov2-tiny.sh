
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
if [ ! -f $VBX_SDK/tutorials/coco2017_rgb_norm_20x416x416x3.npy ]; then
    generate_npy $VBX_SDK/tutorials/coco2017_rgb_20x416x416x3.npy -o $VBX_SDK/tutorials/coco2017_rgb_norm_20x416x416x3.npy -s 416 416  --norm 
fi

echo "Downloading yolov2-tiny..."
# model details @ https://pjreddie.com/darknet/yolo/
[ -f yolov2-tiny.cfg ] || wget -q https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov2-tiny.cfg
[ -f yolov2-tiny.weights ] || wget -q http://web.archive.org/web/20220325031036/https://pjreddie.com/media/files/yolov2-tiny.weights
[ -f coco.names ] || wget -q https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names
python $VBX_SDK/tutorials/darknet/darknet_to_onnx.py yolov2-tiny.cfg

echo "Running ONNX2TF..."
onnx2tf -cind X0 $VBX_SDK/tutorials/coco2017_rgb_norm_20x416x416x3.npy [[[[0.,0.,0.]]]] [[[[1.,1.,1.]]]] \
-b 1 \
-i yolov2-tiny.onnx \
--output_signaturedefs \
--output_integer_quantized_tflite
cp saved_model/yolov2-tiny_full_integer_quant.tflite yolov2-tiny.tflite

if [ -f yolov2-tiny.tflite ]; then
   tflite_preprocess yolov2-tiny.tflite  --scale 255
fi

if [ -f yolov2-tiny.pre.tflite ]; then
    echo "Generating VNNX for V1000 configuration..."
    vnnx_compile -c V1000 -t yolov2-tiny.pre.tflite -o yolov2-tiny.vnnx
fi

if [ -f yolov2-tiny.vnnx ]; then
    echo "Running Simulation..."
    python $VBX_SDK/example/python/yoloInfer.py yolov2-tiny.vnnx $VBX_SDK/tutorials/test_images/dog.jpg -j yolov2-tiny.json -v 2 -l coco.names 
    echo "C Simulation Command:"
    echo '$VBX_SDK/example/sim-c/sim-run-model yolov2-tiny.vnnx $VBX_SDK/tutorials/test_images/dog.jpg YOLOV2'
fi

deactivate
