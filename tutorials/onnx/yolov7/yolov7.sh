
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

echo "Checking for yolov7 files..."

# model details @ https://github.com/ramonhollands/YOLO
[ -f coco.names ] || wget -q https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names
if [ ! -f yolov7.tflite ]; then
   if [ ! -f v7.onnx ]; then
       git clone -b add-export-task https://github.com/ramonhollands/YOLO
       cp yolo.patch YOLO/yolo.patch 
       cd YOLO 
       git apply yolo.patch
       cd ..
       python3 -m venv yolo_venv
       source yolo_venv/bin/activate
       pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
       pip install lightning hydra-core requests rich pillow einops wandb pycocotools onnx
       cd YOLO && python yolo/lazy.py task=export model=v7 task.format=onnx && cd ..
       deactivate
       source $VBX_SDK/vbx_env/bin/activate

       cp YOLO/v7.onnx v7.onnx
   fi
fi


if [ ! -f yolov7.tflite ]; then
   echo "Running ONNX2TF..."
   onnx2tf -cind input $VBX_SDK/tutorials/coco2017_rgb_norm_20x640x640x3.npy [[[[0.,0.,0.]]]] [[[[1.,1.,1.]]]] \
-dgc \
-i v7.onnx \
--output_signaturedefs \
--output_integer_quantized_tflite
   cp saved_model/v7_full_integer_quant.tflite yolov7.tflite
fi
if [ -f yolov7.tflite ]; then
   tflite_preprocess yolov7.tflite  --scale 255
fi

if [ -f yolov7.pre.tflite ]; then
    echo "Generating VNNX for V1000 configuration..."
    vnnx_compile -c V1000 -t yolov7.pre.tflite -o yolov7.vnnx
fi

if [ -f yolov7.vnnx ]; then
    echo "Running Simulation..."
    python $VBX_SDK/example/python/yoloInfer.py yolov7.vnnx $VBX_SDK/tutorials/test_images/dog.jpg -v 7 -t 0.3 -j yolov5n.json -l coco.names 
    echo "C Simulation Command:"
    echo '$VBX_SDK/example/sim-c/sim-run-model yolov7.vnnx $VBX_SDK/tutorials/test_images/dog.jpg YOLOV5'
fi

deactivate
