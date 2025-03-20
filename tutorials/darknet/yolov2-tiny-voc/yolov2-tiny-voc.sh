
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
    wget -P $VBX_SDK/tutorials/ https://vector-blox-model-zoo.s3.us-west-2.amazonaws.com/EAP/calib_npy/coco2017_rgb_norm_20x416x416x3.npy
fi

echo "Downloading yolov2-tiny-voc..."
# model details @ https://pjreddie.com/darknet/yolo/
[ -f yolov2-tiny-voc.cfg ] || wget -q https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov2-tiny-voc.cfg
[ -f yolov2-tiny-voc.weights ] || wget -q http://web.archive.org/web/20220320120309/https://pjreddie.com/media/files/yolov2-tiny-voc.weights
[ -f voc.names ] || wget -q https://raw.githubusercontent.com/pjreddie/darknet/master/data/voc.names
python $VBX_SDK/tutorials/darknet/darknet_to_onnx.py yolov2-tiny-voc.cfg
if [ ! -f calibration_image_sample_data_20x128x128x3_float32.npy ]; then
    wget https://vector-blox-model-zoo.s3.us-west-2.amazonaws.com/EAP/calib_npy/calibration_image_sample_data_20x128x128x3_float32.npy
fi

echo "Running ONNX2TF..."
onnx2tf -cind X0 $VBX_SDK/tutorials/coco2017_rgb_norm_20x416x416x3.npy [[[[0.,0.,0.]]]] [[[[1.,1.,1.]]]] \
-b 1 \
-i yolov2-tiny-voc.onnx \
--output_signaturedefs \
--output_integer_quantized_tflite
cp saved_model/yolov2-tiny-voc_full_integer_quant.tflite yolov2-tiny-voc.tflite

if [ -f yolov2-tiny-voc.tflite ]; then
   tflite_preprocess yolov2-tiny-voc.tflite  --scale 255
fi

if [ -f yolov2-tiny-voc.pre.tflite ]; then
    echo "Generating VNNX for V1000 configuration..."
    vnnx_compile -c V1000 -t yolov2-tiny-voc.pre.tflite -o yolov2-tiny-voc.vnnx
fi

if [ -f yolov2-tiny-voc.vnnx ]; then
    echo "Running Simulation..."
    python $VBX_SDK/example/python/yoloInfer.py yolov2-tiny-voc.vnnx $VBX_SDK/tutorials/test_images/dog.jpg -j yolov2-tiny-voc.json -v 2 -l voc.names 
    echo "C Simulation Command:"
    echo '$VBX_SDK/example/sim-c/sim-run-model yolov2-tiny-voc.vnnx $VBX_SDK/tutorials/test_images/dog.jpg YOLOV2'
fi

deactivate
