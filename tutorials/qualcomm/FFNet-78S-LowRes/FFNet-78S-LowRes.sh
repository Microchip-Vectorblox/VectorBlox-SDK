
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
if [ ! -f $VBX_SDK/tutorials/coco2017_rgb_norm_20x512x1024x3.npy ]; then
    generate_npy $VBX_SDK/tutorials/coco2017_rgb_20x416x416x3.npy -o $VBX_SDK/tutorials/coco2017_rgb_norm_20x512x1024x3.npy -s 512 1024  --norm 
fi

echo "Checking for FFNet-78S-LowRes files..."
if [ ! -f FFNet-78S-LowRes.tflite ]; then
# model details @ 
[ -f FFNet-78S-LowRes.onnx ] || wget -q --no-check-certificate https://huggingface.co/qualcomm/FFNet-78S-LowRes/resolve/97c8b73201f973dffdb4588e881c4dec786d63f3/FFNet-78S-LowRes.onnx
    if [ ! -f calibration_image_sample_data_20x128x128x3_float32.npy ]; then
        wget -q --no-check-certificate https://github.com/Microchip-Vectorblox/assets/raw/refs/heads/main/npy_files/calibration_image_sample_data_20x128x128x3_float32.npy
    fi
fi

if [ -f FFNet-78S-LowRes.onnx ]; then
   if ! echo "3727d640f235ba82fba35dd5bf2d0c8a FFNet-78S-LowRes.onnx" | md5sum -c; then
       echo -e "\n There is an issue with the FFNet-78S-LowRes model file as the expected checksum does not match.\n The model source can be found at: .\n If the model information has changed, please update this script and re-run the tutorial."
       exit 1
   fi
fi


# onnx2tf is an external model conversion tool to convert an onnx model to int8 tflite
# specific operation information can be found here: https://pypi.org/project/onnx2tf/
#  Purpose: Convert source model to int8 tflite format
#  - Required Inputs: onnx compliant model, calibration npy array
#  - Output: int8 tflite model
if [ ! -f FFNet-78S-LowRes.tflite ]; then
   echo "Running ONNX2TF..."
   onnx2tf -cind image $VBX_SDK/tutorials/coco2017_rgb_norm_20x512x1024x3.npy [[[0.,0.,0.]]] [[[1.,1.,1.]]] \
-i FFNet-78S-LowRes.onnx \
--output_signaturedefs \
--output_integer_quantized_tflite
   cp saved_model/FFNet-78S-LowRes_full_integer_quant.tflite FFNet-78S-LowRes.tflite
fi

# tflite_preprocess is an internal tool used to add a preprocess layer to the start of the model
#  Purpose: adds a preprocess layer to the start of the model (if none, will just preprocess by adding a uint8->int8 layer)
#  - Required Inputs: tflite source model, additional arguments 
#  - Outputs: preprocessed tflite model
if [ -f FFNet-78S-LowRes.tflite ]; then
   tflite_preprocess FFNet-78S-LowRes.tflite  --scale 255
fi


# tflite_postprocess is an internal tool that adds a postprocess layer near the end of the model
#  Purpose: adds a postprocess layer at the end of the model
#  - Required Inputs: tflite source model, additional postprocessing arguments
#  - Outputs: tflite model with postprocessed layer at end
if [ -f FFNet-78S-LowRes.pre.tflite ]; then
   tflite_postprocess FFNet-78S-LowRes.pre.tflite  --post-process-layer PIXEL_CITYSCAPES \
--opacity 0.8 \
--height 1080 \
--width 1920
fi


# vnnx_compile is an internal tool that converts an int8 tflite file to a binary file that can be run on the SDK and VectorBlox FPGA
#  Purpose: converts int8 tflite to binary
#  - Required Inputs: int8 tflite, size configuration, compression configuration, output file name
#  - Outputs: binary object files(.hex and binary file)
if [ -f FFNet-78S-LowRes.pre.post.tflite ]; then
    echo "Generating VNNX for V1000 ncomp configuration..."
    vnnx_compile -s V1000 -c ncomp -t FFNet-78S-LowRes.pre.post.tflite  -o FFNet-78S-LowRes_V1000_ncomp.vnnx
fi


# This step runs the final compiled binary in Python, it also shows how to run the same file in C simulation for SDK
#   *Currently C simulation is not supported for unstructured compression
if [ -f FFNet-78S-LowRes_V1000_ncomp.vnnx ]; then
    echo "Running Simulation..."
    python $VBX_SDK/example/python/segmentation.py FFNet-78S-LowRes_V1000_ncomp.vnnx $VBX_SDK/tutorials/test_images/A0PQ76.jpg --dataset cityscapes --inj 
    echo "C Simulation Command:"
    echo '$VBX_SDK/example/sim-c/sim-run-model FFNet-78S-LowRes_V1000_ncomp.vnnx $VBX_SDK/tutorials/test_images/A0PQ76.jpg  '
fi

deactivate
