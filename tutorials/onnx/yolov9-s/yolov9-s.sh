
##########################################################
#  _    __          __             ____  __              #
# | |  / /__  _____/ /_____  _____/ __ )/ /___  _  __    #
# | | / / _ \/ ___/ __/ __ \/ ___/ __  / / __ \| |/_/    #
# | |/ /  __/ /__/ /_/ /_/ / /  / /_/ / / /_/ />  <      #
# |___/\___/\___/\__/\____/_/  /_____/_/\____/_/|_|      #
#                                                        #
# https://github.com/Microchip-Vectorblox/VectorBlox-SDK #
# v3.1                                                   #
#                                                        #
##########################################################



set -e
echo "Checking and activating VBX Python Environment..."
if [ -z $VBX_SDK ]; then
    echo "\$VBX_SDK not set. Please run 'source setup_vars.sh' from the SDK's root folder" && exit 1
fi
source $VBX_SDK/vbx_env/bin/activate


# generate_npy is an internal tool that creates a npy array
#  Purpose: Generates a npy array if an existing one does not exist, this is using custom img data
#  - Required Inputs: source dataset, output name, size
#  - Output: npy array
echo "Checking for Numpy calibration data file..."
if [ ! -f $VBX_SDK/tutorials/coco2017_rgb_norm_20x640x640x3.npy ]; then
    generate_npy $VBX_SDK/tutorials/coco2017_rgb_20x416x416x3.npy -o $VBX_SDK/tutorials/coco2017_rgb_norm_20x640x640x3.npy -s 640 640  --norm 
fi

echo "Checking for yolov9-s files..."

# model details @ https://github.com/MultimediaTechLab/YOLO
[ -f coco.names ] || wget -q https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names
if [ ! -f v9-s.onnx ]; then
    [ -d YOLO ] || git clone https://github.com/MultimediaTechLab/YOLO
    if [ -d YOLO ]; then
       python3 -m venv yolo_venv
       source yolo_venv/bin/activate
       pip install torch==2.8.0 torchvision==0.23 typing_extensions==4.15.0 --index-url https://download.pytorch.org/whl/cpu
       pip install lightning hydra-core requests rich pillow einops wandb pycocotools onnx onnxscript
       cd YOLO
       echo "task: export" > yolo/config/task/export.yaml
       git apply ../lazy.export.diff
       python yolo/lazy.py task=export model=v9-s image_size=[640,640]
       cd ..
       deactivate
       cp YOLO/weights/v9-s.onnx .
       source $VBX_SDK/vbx_env/bin/activate
    fi
fi

if [ ! -f v9-s.sim.onnx ]; then
   onnxsim v9-s.onnx v9-s.sim.onnx
fi

if [ ! -f v9-s.cut.onnx ]; then
python - << EOF
import onnx
outputs = ["1_class", "4_class", "7_class",
"/model/model.22/heads.0/anchor_conv/anchor_conv.2/Conv_output_0",
"/model/model.22/heads.1/anchor_conv/anchor_conv.2/Conv_output_0",
"/model/model.22/heads.2/anchor_conv/anchor_conv.2/Conv_output_0"]
onnx.utils.extract_model("v9-s.sim.onnx", "v9-s.cut.onnx", ["images"], outputs)
EOF
fi

    if [ ! -f calibration_image_sample_data_20x128x128x3_float32.npy ]; then
        wget -q --no-check-certificate https://github.com/Microchip-Vectorblox/assets/raw/refs/heads/main/npy_files/calibration_image_sample_data_20x128x128x3_float32.npy
    fi


if [ -f v9-s.cut.onnx ]; then
   if ! echo "9db1dad64191e4c0b5d3eac5e4238b87 v9-s.cut.onnx" | md5sum -c; then
       echo -e "\n There is an issue with the yolov9-s model file as the expected checksum does not match.\n The model source can be found at: https://github.com/MultimediaTechLab/YOLO.\n If the model information has changed, please update this script and re-run the tutorial."
       exit 1
   fi
fi


# onnx2tf is an external model conversion tool to convert an onnx model to int8 tflite
# specific operation information can be found here: https://pypi.org/project/onnx2tf/
#  Purpose: Convert source model to int8 tflite format
#  - Required Inputs: onnx compliant model, calibration npy array
#  - Output: int8 tflite model
if [ ! -f yolov9-s.tflite ]; then
   echo "Running ONNX2TF..."
   onnx2tf -cind images $VBX_SDK/tutorials/coco2017_rgb_norm_20x640x640x3.npy [[[[0.,0.,0.]]]] [[[[1.,1.,1.]]]] \
-dgc \
-i v9-s.cut.onnx \
--output_signaturedefs \
--output_integer_quantized_tflite
   cp saved_model/v9-s.cut_full_integer_quant.tflite yolov9-s.tflite
fi

# tflite_preprocess is an internal tool used to add a preprocess layer to the start of the model
#  Purpose: adds a preprocess layer to the start of the model (if none, will just preprocess by adding a uint8->int8 layer)
#  - Required Inputs: tflite source model, additional arguments 
#  - Outputs: preprocessed tflite model
if [ -f yolov9-s.tflite ]; then
   tflite_preprocess yolov9-s.tflite  --scale 255
fi


# vnnx_compile is an internal tool that converts an int8 tflite file to a binary file that can be run on the SDK and VectorBlox FPGA
#  Purpose: converts int8 tflite to binary
#  - Required Inputs: int8 tflite, size configuration, compression configuration, output file name
#  - Outputs: binary object files(.hex and binary file)
if [ -f yolov9-s.pre.tflite ]; then
    echo "Generating VNNX for V1000 ncomp configuration..."
    vnnx_compile -s V1000 -c ncomp -t yolov9-s.pre.tflite  -o yolov9-s_V1000_ncomp.vnnx
fi


# This step runs the final compiled binary in Python, it also shows how to run the same file in C simulation for SDK
#   *Currently C simulation is not supported for unstructured compression
if [ -f yolov9-s_V1000_ncomp.vnnx ]; then
    echo "Running Simulation..."
    python $VBX_SDK/example/python/yoloInfer.py yolov9-s_V1000_ncomp.vnnx $VBX_SDK/tutorials/test_images/dog.jpg -v 8 -t 0.3 -l coco.names 
    echo "C Simulation Command:"
    echo '$VBX_SDK/example/sim-c/sim-run-model yolov9-s_V1000_ncomp.vnnx $VBX_SDK/tutorials/test_images/dog.jpg OBJECT_DETECT'
fi

deactivate
