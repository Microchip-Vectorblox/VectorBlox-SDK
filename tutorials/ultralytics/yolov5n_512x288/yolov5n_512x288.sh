
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


# Ultralytics YOLO models are available under the AGPL-3.0 open-source license.
# Projects that are not open source require an Ultralytics Enterprise License. To
# obtain a commercial license for R&D and production use without open-source obligations,
# please complete the licensing form at https://www.ultralytics.com/license.
    

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
if [ ! -f $VBX_SDK/tutorials/coco2017_rgb_norm_20x288x512x3.npy ]; then
    generate_npy $VBX_SDK/tutorials/coco2017_rgb_20x416x416x3.npy -o $VBX_SDK/tutorials/coco2017_rgb_norm_20x288x512x3.npy -s 288 512  --norm 
fi

echo "Checking for yolov5n_512x288 files..."

# model details @ https://github.com/ultralytics/yolov5
[ -f coco.names ] || wget -q https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names
if [ ! -f yolov5n_512x288.tflite ]; then
   [ -f yolov5n.pt ] || wget -q https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5n.pt
   [ -d yolov5 ] || git clone --branch v7.0 https://github.com/ultralytics/yolov5
   if [ -d yolov5 ]; then
       cd yolov5
       python3 -m venv ultralytics
       source ultralytics/bin/activate
       pip install --upgrade pip
       pip install -r requirements.txt
       pip install onnx
       pip install torch==2.5.0 torchvision==0.20.0
       pip uninstall opencv-python -y
       pip install opencv-python-headless
       python export.py --weights ../yolov5n.pt --include onnx --imgsz 288 512
       cd ..
       source $VBX_SDK/vbx_env/bin/activate
   fi
fi
    if [ ! -f calibration_image_sample_data_20x128x128x3_float32.npy ]; then
        wget -q --no-check-certificate https://github.com/Microchip-Vectorblox/assets/raw/refs/heads/main/npy_files/calibration_image_sample_data_20x128x128x3_float32.npy
    fi


if [ -f yolov5n.pt ]; then
   if ! echo "2129d61b0b0034b7c4930140bf2249ea yolov5n.pt" | md5sum -c; then
       echo -e "\n There is an issue with the yolov5n_512x288 model file as the expected checksum does not match.\n The model source can be found at: https://github.com/ultralytics/yolov5.\n If the model information has changed, please update this script and re-run the tutorial."
       exit 1
   fi
fi


# onnx2tf is an external model conversion tool to convert an onnx model to int8 tflite
# specific operation information can be found here: https://pypi.org/project/onnx2tf/
#  Purpose: Convert source model to int8 tflite format
#  - Required Inputs: onnx compliant model, calibration npy array
#  - Output: int8 tflite model
if [ ! -f yolov5n_512x288.tflite ]; then
   echo "Running ONNX2TF..."
   onnx2tf -cind images $VBX_SDK/tutorials/coco2017_rgb_norm_20x288x512x3.npy [[[[0.,0.,0.]]]] [[[[1.,1.,1.]]]] \
--output_names_to_interrupt_model_conversion "/model.24/m.0/Conv_output_0" "/model.24/m.1/Conv_output_0" "/model.24/m.2/Conv_output_0" \
-i yolov5n.onnx \
--output_signaturedefs \
--output_integer_quantized_tflite
   cp saved_model/yolov5n_full_integer_quant.tflite yolov5n_512x288.tflite
fi

# tflite_preprocess is an internal tool used to add a preprocess layer to the start of the model
#  Purpose: adds a preprocess layer to the start of the model (if none, will just preprocess by adding a uint8->int8 layer)
#  - Required Inputs: tflite source model, additional arguments 
#  - Outputs: preprocessed tflite model
if [ -f yolov5n_512x288.tflite ]; then
   tflite_preprocess yolov5n_512x288.tflite  --scale 255
fi


# vnnx_compile is an internal tool that converts an int8 tflite file to a binary file that can be run on the SDK and VectorBlox FPGA
#  Purpose: converts int8 tflite to binary
#  - Required Inputs: int8 tflite, size configuration, compression configuration, output file name
#  - Outputs: binary object files(.hex and binary file)
if [ -f yolov5n_512x288.pre.tflite ]; then
    echo "Generating VNNX for V1000 ncomp configuration..."
    vnnx_compile -s V1000 -c ncomp -t yolov5n_512x288.pre.tflite  -o yolov5n_512x288_V1000_ncomp.vnnx
fi


# This step runs the final compiled binary in Python, it also shows how to run the same file in C simulation for SDK
#   *Currently C simulation is not supported for unstructured compression
if [ -f yolov5n_512x288_V1000_ncomp.vnnx ]; then
    echo "Running Simulation..."
    python $VBX_SDK/example/python/yoloInfer.py yolov5n_512x288_V1000_ncomp.vnnx $VBX_SDK/tutorials/test_images/dog.jpg -j yolov5n.json -v 5 -l coco.names -t 0.25 
    echo "C Simulation Command:"
    echo '$VBX_SDK/example/sim-c/sim-run-model yolov5n_512x288_V1000_ncomp.vnnx $VBX_SDK/tutorials/test_images/dog.jpg YOLOV5'
fi

deactivate
