
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

echo "Downloading yolov5m.relu..."
# model details @ https://github.com/ultralytics/yolov5
[ -f coco.names ] || wget -q https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names
wget -q https://vector-blox-model-zoo.s3.us-west-2.amazonaws.com/Releases/ModelZoo/ultralytics.yolov5m.relu.onnx
if [ ! -f calibration_image_sample_data_20x128x128x3_float32.npy ]; then
    wget https://vector-blox-model-zoo.s3.us-west-2.amazonaws.com/EAP/calib_npy/calibration_image_sample_data_20x128x128x3_float32.npy
fi

echo "Running ONNX2TF..."
onnx2tf -cind images $VBX_SDK/tutorials/coco2017_rgb_norm_20x416x416x3.npy [[[[0.,0.,0.]]]] [[[[1.,1.,1.]]]] \
--overwrite_input_shape images:1,3,416,416 \
--output_names_to_interrupt_model_conversion "onnx::Reshape_367" "onnx::Reshape_405" "onnx::Reshape_443" \
-i ultralytics.yolov5m.relu.onnx \
--output_signaturedefs \
--output_integer_quantized_tflite
cp saved_model/ultralytics.yolov5m.relu_full_integer_quant.tflite yolov5m.relu.tflite

if [ -f yolov5m.relu.tflite ]; then
   tflite_preprocess yolov5m.relu.tflite  --scale 255
fi

if [ -f yolov5m.relu.pre.tflite ]; then
    echo "Generating VNNX for V1000 configuration..."
    vnnx_compile -c V1000 -t yolov5m.relu.pre.tflite -o yolov5m.relu.vnnx
fi

if [ -f yolov5m.relu.vnnx ]; then
    echo "Running Simulation..."
    python $VBX_SDK/example/python/yoloInfer.py yolov5m.relu.vnnx $VBX_SDK/tutorials/test_images/dog.jpg -j yolov5m.json -v 5 -l coco.names -t 0.25 
    echo "C Simulation Command:"
    echo '$VBX_SDK/example/sim-c/sim-run-model yolov5m.relu.vnnx $VBX_SDK/tutorials/test_images/dog.jpg YOLOV5'
fi

deactivate
