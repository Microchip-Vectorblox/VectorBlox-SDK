
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
if [ ! -f $VBX_SDK/tutorials/sample_vehicles_20x288x1024x3.npy ]; then
    wget -P $VBX_SDK/tutorials/ https://vector-blox-model-zoo.s3.us-west-2.amazonaws.com/EAP/calib_npy/sample_vehicles_20x288x1024x3.npy
fi

echo "Downloading lpd_eu_v42..."
# model details @ pytorch/lpd_eu_v42/README.md
[ -f lpd_eu_v42.onnx ] || wget https://vector-blox-model-zoo.s3.us-west-2.amazonaws.com/Releases/ModelZoo/lpd_eu_v42.onnx

echo "Running Model Optimizer..."
mo --input_model lpd_eu_v42.onnx \
--scale_values [255.] \
--static_shape \
--output=Output_Str32_Shape,Output_Str32_Obj,Output_Str16_Shape,Output_Str16_Obj,Output_Str8_Shape,Output_Str8_Obj

echo "Running OpenVINO2Tensorflow..."
openvino2tensorflow --load_dest_file_path_for_the_calib_npy $VBX_SDK/tutorials/sample_vehicles_20x288x1024x3.npy \
--model_path lpd_eu_v42.xml \
--output_full_integer_quant_tflite \
--string_formulas_for_normalization '(data - [0.,0.,0.]) / [1.,1.,1.]'
cp saved_model/model_full_integer_quant.tflite lpd_eu_v42.tflite

if [ -f lpd_eu_v42.tflite ]; then
   tflite_preprocess lpd_eu_v42.tflite   
fi

if [ -f lpd_eu_v42.pre.tflite ]; then
    echo "Generating VNNX for V1000 configuration..."
    vnnx_compile -c V1000 -t lpd_eu_v42.pre.tflite -o lpd_eu_v42.vnnx
fi

if [ -f lpd_eu_v42.vnnx ]; then
    echo "Running Simulation..."
    python $VBX_SDK/example/python/detect_plate.py lpd_eu_v42.vnnx $VBX_SDK/tutorials/test_images/parked_cars.png 
fi

deactivate
