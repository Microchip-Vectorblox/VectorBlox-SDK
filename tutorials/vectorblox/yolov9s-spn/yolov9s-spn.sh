
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
if [ ! -f $VBX_SDK/tutorials/spin_mono_norm_20x512x768x3.npy ]; then
    wget -q --no-check-certificate https://github.com/Microchip-Vectorblox/assets/releases/download/assets/spin_mono_norm_20x512x768x3.npy -O $VBX_SDK/tutorials/spin_mono_norm_20x512x768x3.npy
fi

echo "Checking for yolov9s-spn files..."

# model details @ https://github.com/tpark94/spnv2
if [ ! -f yolov9s-spn.tflite ]; then
   if [ ! -f yolov9s-spn.onnx ]; then
       wget -q --no-check-certificate  https://github.com/Microchip-Vectorblox/assets/releases/download/assets/yolov9s-spn.onnx
   fi

if [ ! -f yolov9s-spn.cut.onnx ]; then
python - <<EOF
import onnx
model_inputs = ['x']
model_outputs = ['conv2d_211', 'conv2d_214', 'conv2d_217',
'conv2d_220', 'conv2d_223', 'conv2d_226',
'conv2d_229', 'conv2d_232', 'conv2d_235']
onnx.utils.extract_model('yolov9s-spn.onnx', 'yolov9s-spn.cut.onnx', model_inputs, model_outputs)
EOF
fi

if [ ! -f yolov9s-spn.opt.onnx ]; then
       onnxsim yolov9s-spn.cut.onnx yolov9s-spn.opt.onnx
   fi
fi
    if [ ! -f calibration_image_sample_data_20x128x128x3_float32.npy ]; then
        wget -q --no-check-certificate https://github.com/Microchip-Vectorblox/assets/raw/refs/heads/main/npy_files/calibration_image_sample_data_20x128x128x3_float32.npy
    fi



# onnx2tf is an external model conversion tool to convert an onnx model to int8 tflite
# specific operation information can be found here: https://pypi.org/project/onnx2tf/
#  Purpose: Convert source model to int8 tflite format
#  - Required Inputs: onnx compliant model, calibration npy array
#  - Output: int8 tflite model
if [ ! -f yolov9s-spn.tflite ]; then
   echo "Running ONNX2TF..."
   onnx2tf -cind x $VBX_SDK/tutorials/spin_mono_norm_20x512x768x3.npy [[[[0.,0.,0.]]]] [[[[1.,1.,1.]]]] \
-dgc \
-ois x:1,3,512,768 \
-i yolov9s-spn.opt.onnx \
--output_signaturedefs \
--output_integer_quantized_tflite
   cp saved_model/yolov9s-spn.opt_full_integer_quant.tflite yolov9s-spn.tflite
fi

# tflite_preprocess is an internal tool used to add a preprocess layer to the start of the model
#  Purpose: adds a preprocess layer to the start of the model (if none, will just preprocess by adding a uint8->int8 layer)
#  - Required Inputs: tflite source model, additional arguments 
#  - Outputs: preprocessed tflite model
if [ -f yolov9s-spn.tflite ]; then
   tflite_preprocess yolov9s-spn.tflite  --scale 255
fi


# vnnx_compile is an internal tool that converts an int8 tflite file to a binary file that can be run on the SDK and VectorBlox FPGA
#  Purpose: converts int8 tflite to binary
#  - Required Inputs: int8 tflite, size configuration, compression configuration, output file name
#  - Outputs: binary object files(.hex and binary file)
if [ -f yolov9s-spn.pre.tflite ]; then
    echo "Generating VNNX for V1000 ncomp configuration..."
    vnnx_compile -s V1000 -c ncomp -t yolov9s-spn.pre.tflite  -o yolov9s-spn_V1000_ncomp.vnnx
fi


# This step runs the final compiled binary in Python, it also shows how to run the same file in C simulation for SDK
#   *Currently C simulation is not supported for unstructured compression
if [ -f yolov9s-spn_V1000_ncomp.vnnx ]; then
    echo "Running Simulation..."
    python $VBX_SDK/example/python/space.py yolov9s-spn_V1000_ncomp.vnnx $VBX_SDK/tutorials/test_images/spin.mono.img000019.jpg 
    echo "C Simulation Command:"
    echo '$VBX_SDK/example/sim-c/sim-run-model yolov9s-spn_V1000_ncomp.vnnx $VBX_SDK/tutorials/test_images/spin.mono.img000019.jpg SPACE'
fi

deactivate
