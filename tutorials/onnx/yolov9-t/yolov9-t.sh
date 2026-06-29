
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

echo "Checking for yolov9-t files..."

# model details @ https://github.com/LibreYOLO/libreyolo
[ -f coco.names ] || wget -q https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names
if [ ! -f v9-t.onnx ]; then
    [ -d libreyolo ] || git clone https://github.com/LibreYOLO/libreyolo
    if [ -d libreyolo ]; then
       python3 -m venv yolo_venv
       source yolo_venv/bin/activate
       pip install torch==2.8.0 torchvision==0.23 typing_extensions==4.15.0 --index-url https://download.pytorch.org/whl/cpu
       pip install lightning hydra-core requests rich pillow einops wandb pycocotools onnx onnxscript scipy
       cd libreyolo
       pip install -e .
       pip uninstall opencv-python opencv-contrib-python -y
       pip install opencv-python-headless
       python - << EOF
from libreyolo import LibreYOLO
model = LibreYOLO("LibreYOLO9t.pt")
model.export(format="onnx", output_path="v9-t.onnx")
EOF
       deactivate
       cd ..
       cp libreyolo/v9-t.onnx .
       source $VBX_SDK/vbx_env/bin/activate
    fi
fi

if [ ! -f v9-t.sim.onnx ]; then
   onnxsim v9-t.onnx v9-t.sim.onnx --overwrite-input-shape "1,3,640,640"
fi

if [ ! -f v9-t.cut.onnx ]; then
python - << EOF
import onnx
outputs = ["/head/cv2.0/cv2.0.2/Conv_output_0",
"/head/cv3.0/cv3.0.2/Conv_output_0",
"/head/cv2.1/cv2.1.2/Conv_output_0",
"/head/cv3.1/cv3.1.2/Conv_output_0",
"/head/cv2.2/cv2.2.2/Conv_output_0",
"/head/cv3.2/cv3.2.2/Conv_output_0"]
onnx.utils.extract_model("v9-t.sim.onnx", "v9-t.cut.onnx", ["images"], outputs)
EOF
fi

    if [ ! -f calibration_image_sample_data_20x128x128x3_float32.npy ]; then
        wget -q --no-check-certificate https://github.com/Microchip-Vectorblox/assets/raw/refs/heads/main/npy_files/calibration_image_sample_data_20x128x128x3_float32.npy
    fi


if [ -f v9-t.cut.onnx ]; then
   if ! echo "7a83b1343264f339e7856739eae90ebc v9-t.cut.onnx" | md5sum -c; then
       echo -e "\n There is an issue with the yolov9-t model file as the expected checksum does not match.\n The model source can be found at: https://github.com/LibreYOLO/libreyolo.\n If the model information has changed, please update this script and re-run the tutorial."
       exit 1
   fi
fi


# onnx2tf is an external model conversion tool to convert an onnx model to int8 tflite
# specific operation information can be found here: https://pypi.org/project/onnx2tf/
#  Purpose: Convert source model to int8 tflite format
#  - Required Inputs: onnx compliant model, calibration npy array
#  - Output: int8 tflite model
if [ ! -f yolov9-t.tflite ]; then
   echo "Running ONNX2TF..."
   onnx2tf -cind images $VBX_SDK/tutorials/coco2017_rgb_norm_20x640x640x3.npy [[[[0.,0.,0.]]]] [[[[1.,1.,1.]]]] \
-dgc \
-i v9-t.cut.onnx \
--output_signaturedefs \
--output_integer_quantized_tflite
   cp saved_model/v9-t.cut_full_integer_quant.tflite yolov9-t.tflite
fi

# tflite_preprocess is an internal tool used to add a preprocess layer to the start of the model
#  Purpose: adds a preprocess layer to the start of the model (if none, will just preprocess by adding a uint8->int8 layer)
#  - Required Inputs: tflite source model, additional arguments 
#  - Outputs: preprocessed tflite model
if [ -f yolov9-t.tflite ]; then
   tflite_preprocess yolov9-t.tflite  --scale 255
fi


# vnnx_compile is an internal tool that converts an int8 tflite file to a binary file that can be run on the SDK and VectorBlox FPGA
#  Purpose: converts int8 tflite to binary
#  - Required Inputs: int8 tflite, size configuration, compression configuration, output file name
#  - Outputs: binary object files(.hex and binary file)
if [ -f yolov9-t.pre.tflite ]; then
    echo "Generating VNNX for V1000 ncomp configuration..."
    vnnx_compile -s V1000 -c ncomp -t yolov9-t.pre.tflite  -o yolov9-t_V1000_ncomp.vnnx
fi


# This step runs the final compiled binary in Python, it also shows how to run the same file in C simulation for SDK
#   *Currently C simulation is not supported for unstructured compression
if [ -f yolov9-t_V1000_ncomp.vnnx ]; then
    echo "Running Simulation..."
    python $VBX_SDK/example/python/yoloInfer.py yolov9-t_V1000_ncomp.vnnx $VBX_SDK/tutorials/test_images/dog.jpg -v 8 -t 0.2 -l coco.names 
    echo "C Simulation Command:"
    echo '$VBX_SDK/example/sim-c/sim-run-model yolov9-t_V1000_ncomp.vnnx $VBX_SDK/tutorials/test_images/dog.jpg OBJECT_DETECT'
fi

deactivate
