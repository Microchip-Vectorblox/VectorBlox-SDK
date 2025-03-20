
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
if [ ! -f $VBX_SDK/tutorials/sample_plates_20x34x146x3.npy ]; then
    wget -P $VBX_SDK/tutorials/ https://vector-blox-model-zoo.s3.us-west-2.amazonaws.com/EAP/calib_npy/sample_plates_20x34x146x3.npy
fi

echo "Downloading lpr_eu_v3..."
# model details @ pytorch/lpr_eu_v3/README.md
[ -f lpr_eu_v3.onnx ] || wget -q https://vector-blox-model-zoo.s3.us-west-2.amazonaws.com/Releases/ModelZoo/lpr_eu_v3.onnx

echo "Running Model Optimizer..."
mo --input_model lpr_eu_v3.onnx \
--scale_values [255.] \
--static_shape \
--input_shape [1,3,34,146]

echo "Running OpenVINO2Tensorflow..."
openvino2tensorflow --load_dest_file_path_for_the_calib_npy $VBX_SDK/tutorials/sample_plates_20x34x146x3.npy \
--model_path lpr_eu_v3.xml \
--output_full_integer_quant_tflite \
--string_formulas_for_normalization '(data - [0.,0.,0.]) / [1.,1.,1.]'
cp saved_model/model_full_integer_quant.tflite lpr_eu_v3.tflite

if [ -f lpr_eu_v3.tflite ]; then
   tflite_preprocess lpr_eu_v3.tflite   
fi

if [ -f lpr_eu_v3.pre.tflite ]; then
    echo "Generating VNNX for V1000 configuration..."
    vnnx_compile -c V1000 -t lpr_eu_v3.pre.tflite -o lpr_eu_v3.vnnx
fi

if [ -f lpr_eu_v3.vnnx ]; then
    echo "Running Simulation..."
    python $VBX_SDK/example/python/recognize_plate_eu.py lpr_eu_v3.vnnx $VBX_SDK/tutorials/test_images/A358CC82.jpg 
    echo "C Simulation Command:"
    echo '$VBX_SDK/example/sim-c/sim-run-model lpr_eu_v3.vnnx $VBX_SDK/tutorials/test_images/A358CC82.jpg LPR'
fi

deactivate
