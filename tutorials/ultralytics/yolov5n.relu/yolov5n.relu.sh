
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

echo "Checking for Numpy calibration data file..."
if [ ! -f $VBX_SDK/tutorials/coco2017_rgb_norm_20x416x416x3.npy ]; then
    generate_npy $VBX_SDK/tutorials/coco2017_rgb_20x416x416x3.npy -o $VBX_SDK/tutorials/coco2017_rgb_norm_20x416x416x3.npy -s 416 416  --norm 
fi

echo "Checking for yolov5n.relu files..."

# model details @ https://github.com/ultralytics/yolov5
[ -f coco.names ] || wget -q https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names
if [ ! -f yolov5n.relu.tflite ]; then
   wget -q --no-check-certificate https://github.com/Microchip-Vectorblox/assets/releases/download/assets/ultralytics.yolov5n.relu.onnx
fi


if [ ! -f yolov5n.relu.tflite ]; then
   echo "Running ONNX2TF..."
   onnx2tf -cind images $VBX_SDK/tutorials/coco2017_rgb_norm_20x416x416x3.npy [[[[0.,0.,0.]]]] [[[[1.,1.,1.]]]] \
--overwrite_input_shape images:1,3,416,416 \
--output_names_to_interrupt_model_conversion "335" "401" "269" \
-i ultralytics.yolov5n.relu.onnx \
--output_signaturedefs \
--output_integer_quantized_tflite
   cp saved_model/ultralytics.yolov5n.relu_full_integer_quant.tflite yolov5n.relu.tflite
fi
if [ -f yolov5n.relu.tflite ]; then
   tflite_preprocess yolov5n.relu.tflite  --scale 255
fi

if [ -f yolov5n.relu.pre.tflite ]; then
    echo "Generating VNNX for V1000 ncomp configuration..."
    vnnx_compile -s V1000 -c ncomp -t yolov5n.relu.pre.tflite  -o yolov5n.relu_V1000_ncomp.vnnx
fi

if [ -f yolov5n.relu_V1000_ncomp.vnnx ]; then
    echo "Running Simulation..."
    python $VBX_SDK/example/python/yoloInfer.py yolov5n.relu_V1000_ncomp.vnnx $VBX_SDK/tutorials/test_images/dog.jpg -j yolov5n.json -v 5 -l coco.names -t 0.25 
    echo "C Simulation Command:"
    echo '$VBX_SDK/example/sim-c/sim-run-model yolov5n.relu_V1000_ncomp.vnnx $VBX_SDK/tutorials/test_images/dog.jpg YOLOV5'
fi

deactivate
