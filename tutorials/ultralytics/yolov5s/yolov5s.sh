
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
if [ ! -f $VBX_SDK/tutorials/coco2017_rgb_norm_20x640x640x3.npy ]; then
    generate_npy $VBX_SDK/tutorials/coco2017_rgb_20x416x416x3.npy -o $VBX_SDK/tutorials/coco2017_rgb_norm_20x640x640x3.npy -s 640 640  --norm 
fi

echo "Checking for yolov5s files..."

# model details @ https://github.com/ultralytics/yolov5
[ -f coco.names ] || wget -q https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names
if [ ! -f yolov5s.tflite ]; then
   [ -f yolov5s.pt ] || wget -q https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5s.pt
   [ -d yolov5 ] || git clone --branch v7.0 https://github.com/ultralytics/yolov5
   if [ -d yolov5 ]; then
       cd yolov5
       python3 -m venv ultralytics
       source ultralytics/bin/activate
       pip install --upgrade pip
       pip install -r requirements.txt
       pip install onnx
       pip install torch==2.5.0 torchvision==0.20.0
       python export.py --weights ../yolov5s.pt --include onnx
       cd ..
       source $VBX_SDK/vbx_env/bin/activate
   fi
fi


if [ ! -f yolov5s.tflite ]; then
   echo "Running ONNX2TF..."
   onnx2tf -cind images $VBX_SDK/tutorials/coco2017_rgb_norm_20x640x640x3.npy [[[[0.,0.,0.]]]] [[[[1.,1.,1.]]]] \
--output_names_to_interrupt_model_conversion "/model.24/m.0/Conv_output_0" "/model.24/m.1/Conv_output_0" "/model.24/m.2/Conv_output_0" \
-i yolov5s.onnx \
--output_signaturedefs \
--output_integer_quantized_tflite
   cp saved_model/yolov5s_full_integer_quant.tflite yolov5s.tflite
fi
if [ -f yolov5s.tflite ]; then
   tflite_preprocess yolov5s.tflite  --scale 255
fi

if [ -f yolov5s.pre.tflite ]; then
    echo "Generating VNNX for V1000 configuration..."
    vnnx_compile -c V1000 -t yolov5s.pre.tflite -o yolov5s.vnnx
fi

if [ -f yolov5s.vnnx ]; then
    echo "Running Simulation..."
    python $VBX_SDK/example/python/yoloInfer.py yolov5s.vnnx $VBX_SDK/tutorials/test_images/dog.jpg -j yolov5s.json -v 5 -l coco.names -t 0.25 
    echo "C Simulation Command:"
    echo '$VBX_SDK/example/sim-c/sim-run-model yolov5s.vnnx $VBX_SDK/tutorials/test_images/dog.jpg  '
fi

deactivate
