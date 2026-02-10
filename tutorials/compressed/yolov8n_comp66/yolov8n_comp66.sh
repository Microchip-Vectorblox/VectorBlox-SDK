
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
if [ ! -f $VBX_SDK/tutorials/coco2017_rgb_norm_20x640x640x3.npy ]; then
    generate_npy $VBX_SDK/tutorials/coco2017_rgb_20x416x416x3.npy -o $VBX_SDK/tutorials/coco2017_rgb_norm_20x640x640x3.npy -s 640 640  --norm 
fi

echo "Checking for yolov8n_comp66 files..."

# model details @ https://github.com/ultralytics/ultralytics/
[ -f coco.names ] || wget -q https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names
if [ ! -f yolov8n_comp66.tflite ]; then
   wget -q https://github.com/Microchip-Vectorblox/assets/releases/download/assets/yolov8n_comp66.onnx
fi


if [ ! -f yolov8n_comp66.tflite ]; then
   echo "Running ONNX2TF..."
   onnx2tf -cind images $VBX_SDK/tutorials/coco2017_rgb_norm_20x640x640x3.npy [[[[0.,0.,0.]]]] [[[[1.,1.,1.]]]] \
--disable_group_convolution \
--output_names_to_interrupt_model_conversion "/model.22/cv2.0/cv2.0.2/Conv_output_0" "/model.22/cv3.0/cv3.0.2/Conv_output_0" "/model.22/cv2.1/cv2.1.2/Conv_output_0" "/model.22/cv3.1/cv3.1.2/Conv_output_0" "/model.22/cv2.2/cv2.2.2/Conv_output_0" "/model.22/cv3.2/cv3.2.2/Conv_output_0" \
-i yolov8n_comp66.onnx \
--output_signaturedefs \
--output_integer_quantized_tflite
   cp saved_model/yolov8n_comp66_full_integer_quant.tflite yolov8n_comp66.tflite
fi
if [ -f yolov8n_comp66.tflite ]; then
   tflite_preprocess yolov8n_comp66.tflite  --scale 255
fi

if [ -f yolov8n_comp66.pre.tflite ]; then
    echo "Generating VNNX for V1000 comp configuration..."
    vnnx_compile -s V1000 -c comp -t yolov8n_comp66.pre.tflite  -o yolov8n_comp66_V1000_comp.vnnx
fi

if [ -f yolov8n_comp66_V1000_comp.vnnx ]; then
    echo "Running Simulation..."
    python $VBX_SDK/example/python/yoloInfer.py yolov8n_comp66_V1000_comp.vnnx $VBX_SDK/tutorials/test_images/dog.jpg -v 8 -l coco.names 
    echo "C Simulation Command:"
    echo '$VBX_SDK/example/sim-c/sim-run-model yolov8n_comp66_V1000_comp.vnnx $VBX_SDK/tutorials/test_images/dog.jpg ULTRALYTICS'
fi

deactivate
