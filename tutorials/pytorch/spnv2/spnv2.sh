
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

echo "Checking for spnv2 files..."

# model details @ https://github.com/tpark94/spnv2
echo 'The onnx2tf step takes 2-3+ hrs and vnnx_compile can take up to 1+ hr long'
echo 'For a more optimized solution please see vectorblox/yolov9s-spn tutorial'
if [ ! -f spnv2.tflite ]; then
   if [ ! -f spnv2_efficientnetb3_fullconfig_offline.pth.tar  ]; then
       echo 'Go to https://github.com/tpark94/spnv2 and download pre-trained weights(spnv2_efficientnetb3_fullconfig_offline.pth.tar) from the Optional Step 5 of the Installation'
       echo 'The above tar file should be placed in this directory'
       exit
   fi

if [ ! -f spnv2.onnx ]; then
    git clone https://github.com/tpark94/spnv2
    cd spnv2
    python3.8 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install albumentations==1.4.18 matplotlib==3.4.2 pandas==1.3.1 plyfile==0.7.4 PyYAML==6.0.2 scipy==1.10.0 torch==1.13.1 torchinfo==1.8.0 torchvision==0.14.1 tqdm==4.67.1 yacs==0.1.8 onnx==1.17.0
    cp ../spnv2.export.py .
    python spnv2.export.py
    deactivate
    cd ../
    cp spnv2/spnv2.onnx .
fi

if [ ! -f spnv2.cut.onnx ]; then
python - <<EOF
import onnx
model_inputs = ['input.1']
model_outputs = ['/class_net/head/conv_pw/Conv_output_0',
'/class_net/head/conv_pw_1/Conv_output_0',
'/class_net/head/conv_pw_2/Conv_output_0',
'/class_net/head/conv_pw_3/Conv_output_0',
'/class_net/head/conv_pw_4/Conv_output_0',
'/box_net/head/conv_pw/Conv_output_0',
'/box_net/head/conv_pw_1/Conv_output_0',
'/box_net/head/conv_pw_2/Conv_output_0',
'/box_net/head/conv_pw_3/Conv_output_0',
'/box_net/head/conv_pw_4/Conv_output_0',
'/rotation_net/Add_1_output_0',
'/rotation_net/Add_3_output_0',
'/rotation_net/Add_5_output_0',
'/rotation_net/Add_7_output_0',
'/rotation_net/Add_9_output_0',
'/translation_net/Add_2_output_0',
'/translation_net/Add_3_output_0',
'/translation_net/Add_6_output_0',
'/translation_net/Add_7_output_0',
'/translation_net/Add_10_output_0',
'/translation_net/Add_11_output_0',
'/translation_net/Add_14_output_0',
'/translation_net/Add_15_output_0',
'/translation_net/Add_18_output_0',
'/translation_net/Add_19_output_0']
onnx.utils.extract_model('spnv2.onnx', 'spnv2.cut.onnx', model_inputs, model_outputs)
EOF
fi

   if [ ! -f spnv2.opt.onnx ]; then
       onnxsim spnv2.cut.onnx spnv2.opt.onnx
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
if [ ! -f spnv2.tflite ]; then
   echo "Running ONNX2TF..."
   onnx2tf -cind input.1 $VBX_SDK/tutorials/spin_mono_norm_20x512x768x3.npy [[[[0.485,0.456,0.406]]]] [[[[0.229,0.224,0.225]]]] \
-dgc \
-i spnv2.opt.onnx \
--output_signaturedefs \
--output_integer_quantized_tflite
   cp saved_model/spnv2.opt_full_integer_quant.tflite spnv2.tflite
fi

# tflite_preprocess is an internal tool used to add a preprocess layer to the start of the model
#  Purpose: adds a preprocess layer to the start of the model (if none, will just preprocess by adding a uint8->int8 layer)
#  - Required Inputs: tflite source model, additional arguments 
#  - Outputs: preprocessed tflite model
if [ -f spnv2.tflite ]; then
   tflite_preprocess spnv2.tflite  --scale 58.395 57.12 57.375 --mean 123.675 116.28 103.53
fi


# tflite_transform is an internal tool used to add various custom transformations to the model
# additional information found here: https://github.com/Microchip-Vectorblox/VectorBlox-SDK/blob/master/python/vbx/vbx/generate/transform_tflite.py
# supported arguments are found in the `passes` struct at the top#  Purpose: Adds various transformations to a tflite model
#  - Required Inputs: tflite source model, additional arguments
#  - Outputs: transformed tflite model
if [ -f spnv2.pre.tflite ]; then
   tflite_transform spnv2.pre.tflite  -p FUSE_GROUP1 SPNV2_TRANSPOSE REWRITE_NORM
fi


# vnnx_compile is an internal tool that converts an int8 tflite file to a binary file that can be run on the SDK and VectorBlox FPGA
#  Purpose: converts int8 tflite to binary
#  - Required Inputs: int8 tflite, size configuration, compression configuration, output file name
#  - Outputs: binary object files(.hex and binary file)
if [ -f spnv2.pre.tr.tflite ]; then
    echo "Generating VNNX for V1000 ncomp configuration..."
    vnnx_compile -s V1000 -c ncomp -t spnv2.pre.tr.tflite  -o spnv2_V1000_ncomp.vnnx
fi


# This step runs the final compiled binary in Python, it also shows how to run the same file in C simulation for SDK
#   *Currently C simulation is not supported for unstructured compression
if [ -f spnv2_V1000_ncomp.vnnx ]; then
    echo "Running Simulation..."
    python $VBX_SDK/example/python/space.py spnv2_V1000_ncomp.vnnx $VBX_SDK/tutorials/test_images/spin.mono.img000019.jpg -t 
    echo "C Simulation Command:"
    echo '$VBX_SDK/example/sim-c/sim-run-model spnv2_V1000_ncomp.vnnx $VBX_SDK/tutorials/test_images/spin.mono.img000019.jpg SPACE_T'
fi

deactivate
