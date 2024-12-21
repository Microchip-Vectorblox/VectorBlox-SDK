
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
if [ ! -f $VBX_SDK/tutorials/coco2017_rgb_norm_20x224x224x3.npy ]; then
    wget -P $VBX_SDK/tutorials/ https://vector-blox-model-zoo.s3.us-west-2.amazonaws.com/EAP/calib_npy/coco2017_rgb_norm_20x224x224x3.npy
fi

echo "Downloading yolov8n-cls..."
# model details @ https://github.com/ultralytics/ultralytics/
# This currently has worse accuracy
#if [ ! -f yolov8n-cls.tflite ]; then
#    yolo export model=yolov8n-cls.pt format=tflite int8 || true
#    cp yolov8n-cls_saved_model/yolov8n-cls_full_integer_quant.tflite yolov8n-cls.tflite
#fi
#
if [ ! -f yolov8n-cls.onnx ]; then
    yolo export model=yolov8n-cls.pt format=onnx
fi
if [ ! -f calibration_image_sample_data_20x128x128x3_float32.npy ]; then
    wget https://vector-blox-model-zoo.s3.us-west-2.amazonaws.com/EAP/calib_npy/calibration_image_sample_data_20x128x128x3_float32.npy
fi

echo "Running ONNX2TF..."
onnx2tf -cind images $VBX_SDK/tutorials/coco2017_rgb_norm_20x224x224x3.npy [[[[0.,0.,0.]]]] [[[[1.,1.,1.]]]] \
-i yolov8n-cls.onnx \
--output_signaturedefs \
--output_integer_quantized_tflite
cp saved_model/yolov8n-cls_full_integer_quant.tflite yolov8n-cls.tflite

if [ -f yolov8n-cls.tflite ]; then
   tflite_preprocess yolov8n-cls.tflite  --scale 255
fi

if [ -f yolov8n-cls.pre.tflite ]; then
    echo "Generating VNNX for V1000 configuration..."
    vnnx_compile -c V1000 -t yolov8n-cls.pre.tflite -o yolov8n-cls.vnnx
fi

if [ -f yolov8n-cls.vnnx ]; then
    echo "Running Simulation..."
    python $VBX_SDK/example/python/classifier.py yolov8n-cls.vnnx $VBX_SDK/tutorials/test_images/oreo.jpg 
fi

deactivate
