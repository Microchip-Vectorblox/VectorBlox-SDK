
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
if [ ! -f $VBX_SDK/tutorials/coco2017_rgb_norm_20x288x512x3.npy ]; then
    generate_npy $VBX_SDK/tutorials/coco2017_rgb_20x416x416x3.npy -o $VBX_SDK/tutorials/coco2017_rgb_norm_20x288x512x3.npy -s 288 512  --norm 
fi

echo "Checking for FFNet-122NS-LowRes_512x288 files..."

# model details @ 
if [ ! -f FFNet-122NS-LowRes_512x288.tflite ]; then
   [ -f FFNet-122NS-LowRes.onnx ] || wget -q --no-check-certificate https://huggingface.co/qualcomm/FFNet-122NS-LowRes/resolve/568e7f6ecce9c9df9423b7b834cc90d1bc2b2cbd/FFNet-122NS-LowRes.onnx
python - <<EOF
import onnx
from onnx.tools import update_model_dims
import numpy as np
from onnxsim import simplify
model = onnx.load('FFNet-122NS-LowRes.onnx')
model.graph.input[0].type.tensor_type.shape.dim[2].Clear()
model.graph.input[0].type.tensor_type.shape.dim[3].Clear()
for info in model.graph.value_info:
    n = len(info.type.tensor_type.shape.dim)
    for ind in range(n-2,n):
        info.type.tensor_type.shape.dim[ind].Clear()
model.graph.output[0].type.tensor_type.shape.dim[1].Clear()
model.graph.output[0].type.tensor_type.shape.dim[2].Clear()
for init in model.graph.initializer:
    if init.name == '/model/head/layers/1/Concat_output_0':
        t = onnx.numpy_helper.from_array(np.array([1,128,9,16],dtype='int64'))
        init.raw_data = t.raw_data
    if init.name == '/model/head/layers/1/Concat_1_output_0':
        t = onnx.numpy_helper.from_array(np.array([1,128,18,32],dtype='int64'))
        init.raw_data = t.raw_data
    if init.name == '/model/head/layers/1/Concat_2_output_0':
        t = onnx.numpy_helper.from_array(np.array([1,128,36,64],dtype='int64'))
        init.raw_data = t.raw_data
    if init.name == '/model/head/layers/2/Concat_5_output_0':
        t = onnx.numpy_helper.from_array(np.array([1,16,36,64],dtype='int64'))
        init.raw_data = t.raw_data
model_512x288 = update_model_dims.update_inputs_outputs_dims(model, {"image":[1,3,288,512]},{"mask":[1,288,512]})
model_simp, check = simplify(model_512x288)
onnx.save(model_simp,'FFNet-122NS-LowRes_512x288.onnx')
EOF
fi

    if [ ! -f calibration_image_sample_data_20x128x128x3_float32.npy ]; then
        wget -q --no-check-certificate https://github.com/Microchip-Vectorblox/assets/raw/refs/heads/main/npy_files/calibration_image_sample_data_20x128x128x3_float32.npy
    fi


if [ -f FFNet-122NS-LowRes.onnx ]; then
   if ! echo "489070f04812e87310ee734d31ec4af6 FFNet-122NS-LowRes.onnx" | md5sum -c; then
       echo -e "\n There is an issue with the FFNet-122NS-LowRes_512x288 model file as the expected checksum does not match.\n The model source can be found at: .\n If the model information has changed, please update this script and re-run the tutorial."
       exit 1
   fi
fi


# onnx2tf is an external model conversion tool to convert an onnx model to int8 tflite
# specific operation information can be found here: https://pypi.org/project/onnx2tf/
#  Purpose: Convert source model to int8 tflite format
#  - Required Inputs: onnx compliant model, calibration npy array
#  - Output: int8 tflite model
if [ ! -f FFNet-122NS-LowRes_512x288.tflite ]; then
   echo "Running ONNX2TF..."
   onnx2tf -cind image $VBX_SDK/tutorials/coco2017_rgb_norm_20x288x512x3.npy [[[0.,0.,0.]]] [[[1.,1.,1.]]] \
-i FFNet-122NS-LowRes_512x288.onnx \
--output_signaturedefs \
--output_integer_quantized_tflite
   cp saved_model/FFNet-122NS-LowRes_512x288_full_integer_quant.tflite FFNet-122NS-LowRes_512x288.tflite
fi

# tflite_preprocess is an internal tool used to add a preprocess layer to the start of the model
#  Purpose: adds a preprocess layer to the start of the model (if none, will just preprocess by adding a uint8->int8 layer)
#  - Required Inputs: tflite source model, additional arguments 
#  - Outputs: preprocessed tflite model
if [ -f FFNet-122NS-LowRes_512x288.tflite ]; then
   tflite_preprocess FFNet-122NS-LowRes_512x288.tflite  --scale 255
fi


# tflite_postprocess is an internal tool that adds a postprocess layer near the end of the model
#  Purpose: adds a postprocess layer at the end of the model
#  - Required Inputs: tflite source model, additional postprocessing arguments
#  - Outputs: tflite model with postprocessed layer at end
if [ -f FFNet-122NS-LowRes_512x288.pre.tflite ]; then
   tflite_postprocess FFNet-122NS-LowRes_512x288.pre.tflite  --post-process-layer PIXEL_CITYSCAPES \
--opacity 0.8 \
--height 1080 \
--width 1920
fi


# vnnx_compile is an internal tool that converts an int8 tflite file to a binary file that can be run on the SDK and VectorBlox FPGA
#  Purpose: converts int8 tflite to binary
#  - Required Inputs: int8 tflite, size configuration, compression configuration, output file name
#  - Outputs: binary object files(.hex and binary file)
if [ -f FFNet-122NS-LowRes_512x288.pre.post.tflite ]; then
    echo "Generating VNNX for V1000 ncomp configuration..."
    vnnx_compile -s V1000 -c ncomp -t FFNet-122NS-LowRes_512x288.pre.post.tflite  -o FFNet-122NS-LowRes_512x288_V1000_ncomp.vnnx
fi


# This step runs the final compiled binary in Python, it also shows how to run the same file in C simulation for SDK
#   *Currently C simulation is not supported for unstructured compression
if [ -f FFNet-122NS-LowRes_512x288_V1000_ncomp.vnnx ]; then
    echo "Running Simulation..."
    python $VBX_SDK/example/python/segmentation.py FFNet-122NS-LowRes_512x288_V1000_ncomp.vnnx $VBX_SDK/tutorials/test_images/A0PQ76.jpg --dataset cityscapes --inj 
    echo "C Simulation Command:"
    echo '$VBX_SDK/example/sim-c/sim-run-model FFNet-122NS-LowRes_512x288_V1000_ncomp.vnnx $VBX_SDK/tutorials/test_images/A0PQ76.jpg  '
fi

deactivate
