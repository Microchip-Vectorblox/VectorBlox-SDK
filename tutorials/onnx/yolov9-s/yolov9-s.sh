
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

echo "Checking for yolov9-s files..."

# model details @ https://github.com/ramonhollands/YOLO
[ -f coco.names ] || wget -q https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names
if [ ! -f yolov9-s.tflite ]; then
   if [ ! -f v9-s.cut.onnx ]; then
       git clone -b add-export-task https://github.com/ramonhollands/YOLO
       sed -i '173d;174d;' YOLO/yolo/tools/solver.py
       python3 -m venv yolo_venv
       source yolo_venv/bin/activate
       pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
       pip install lightning hydra-core requests rich pillow einops wandb pycocotools onnx
       cd YOLO && python yolo/lazy.py task=export model=v9-s task.format=onnx && cd ..
       deactivate
       source $VBX_SDK/vbx_env/bin/activate

python - << EOF
import onnx
outputs = ["1_class_scores_small", "4_class_scores_medium", "7_class_scores_large",
"/model.22/heads.0/anchor_conv/anchor_conv.2/Conv_output_0",
"/model.22/heads.1/anchor_conv/anchor_conv.2/Conv_output_0",
"/model.22/heads.2/anchor_conv/anchor_conv.2/Conv_output_0"]
onnx.utils.extract_model("./YOLO/v9-s.onnx", "v9-s.cut.onnx", ["input"], outputs)
EOF

   fi
fi


if [ ! -f yolov9-s.tflite ]; then
   echo "Running ONNX2TF..."
   onnx2tf -cind input $VBX_SDK/tutorials/coco2017_rgb_norm_20x640x640x3.npy [[[[0.,0.,0.]]]] [[[[1.,1.,1.]]]] \
-dgc \
-i v9-s.cut.onnx \
--output_signaturedefs \
--output_integer_quantized_tflite
   cp saved_model/v9-s.cut_full_integer_quant.tflite yolov9-s.tflite
fi
if [ -f yolov9-s.tflite ]; then
   tflite_preprocess yolov9-s.tflite  --scale 255
fi

if [ -f yolov9-s.pre.tflite ]; then
    echo "Generating VNNX for V1000 configuration..."
    vnnx_compile -c V1000 -t yolov9-s.pre.tflite -o yolov9-s.vnnx
fi

if [ -f yolov9-s.vnnx ]; then
    echo "Running Simulation..."
    python $VBX_SDK/example/python/yoloInfer.py yolov9-s.vnnx $VBX_SDK/tutorials/test_images/dog.jpg -v 8 -t 0.3 -l coco.names 
    echo "C Simulation Command:"
    echo '$VBX_SDK/example/sim-c/sim-run-model yolov9-s.vnnx $VBX_SDK/tutorials/test_images/dog.jpg ULTRALYTICS'
fi

deactivate
