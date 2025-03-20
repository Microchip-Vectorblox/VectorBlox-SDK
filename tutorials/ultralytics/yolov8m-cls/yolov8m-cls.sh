
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

echo "Downloading yolov8m-cls..."
# model details @ https://github.com/ultralytics/ultralytics/
if [ ! -f yolov8m-cls.onnx ]; then
    yolo export model=yolov8m-cls.pt format=onnx
fi
if [ ! -f calibration_image_sample_data_20x128x128x3_float32.npy ]; then
    wget https://vector-blox-model-zoo.s3.us-west-2.amazonaws.com/EAP/calib_npy/calibration_image_sample_data_20x128x128x3_float32.npy
fi

echo "Running ONNX2TF..."
onnx2tf -cind images $VBX_SDK/tutorials/coco2017_rgb_norm_20x224x224x3.npy [[[[0.,0.,0.]]]] [[[[1.,1.,1.]]]] \
-i yolov8m-cls.onnx \
--output_signaturedefs \
--output_integer_quantized_tflite
cp saved_model/yolov8m-cls_full_integer_quant.tflite yolov8m-cls.tflite

if [ -f yolov8m-cls.tflite ]; then
   tflite_preprocess yolov8m-cls.tflite  --scale 255
fi

if [ -f yolov8m-cls.pre.tflite ]; then
    echo "Generating VNNX for V1000 configuration..."
    vnnx_compile -c V1000 -t yolov8m-cls.pre.tflite -o yolov8m-cls.vnnx
fi

if [ -f yolov8m-cls.vnnx ]; then
    echo "Running Simulation..."
    python $VBX_SDK/example/python/classifier.py yolov8m-cls.vnnx $VBX_SDK/tutorials/test_images/oreo.jpg 
    echo "C Simulation Command:"
    echo '$VBX_SDK/example/sim-c/sim-run-model yolov8m-cls.vnnx $VBX_SDK/tutorials/test_images/oreo.jpg CLASSIFY'
fi

deactivate
