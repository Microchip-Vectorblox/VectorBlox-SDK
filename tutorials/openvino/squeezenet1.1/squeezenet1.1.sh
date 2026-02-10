
##########################################################
#  _    __          __             ____  __              #
# | |  / /__  _____/ /_____  _____/ __ )/ /___  _  __    #
# | | / / _ \/ ___/ __/ __ \/ ___/ __  / / __ \| |/_/    #
# | |/ /  __/ /__/ /_/ /_/ / /  / /_/ / / /_/ />  <      #
# |___/\___/\___/\__/\____/_/  /_____/_/\____/_/|_|      #
#                                                        #
# https://github.com/Microchip-Vectorblox/VectorBlox-SDK #
# v3.0                                                   #
#                                                        #
##########################################################

set -e
echo "Checking and activating VBX Python Environment..."
if [ -z $VBX_SDK ]; then
    echo "\$VBX_SDK not set. Please run 'source setup_vars.sh' from the SDK's root folder" && exit 1
fi
source $VBX_SDK/vbx_env/bin/activate

echo "Checking for Numpy calibration data file..."
if [ ! -f $VBX_SDK/tutorials/imagenetv2_20x227x227x3.npy ]; then
    generate_npy $VBX_SDK/tutorials/imagenetv2_rgb_20x224x224x3.npy -o $VBX_SDK/tutorials/imagenetv2_20x227x227x3.npy -s 227 227  -b 
fi

echo "Checking for squeezenet1.1 files..."

# model details @ https://github.com/openvinotoolkit/open_model_zoo/tree/2021.4.2/models/public/squeezenet1.1/
if [ ! -f squeezenet1.1.tflite ]; then 
omz_downloader --name squeezenet1.1
fi


if [ ! -f squeezenet1.1.tflite ]; then
   echo "Running Model Optimizer..."
   mo --input_model public/squeezenet1.1/squeezenet1.1.caffemodel \
--reverse_input_channels \
--mean_values [104,117,123] \
--static_shape \
--input_shape [1,3,227,227]
fi
if [ ! -f squeezenet1.1.tflite ]; then
   echo "Running OpenVINO2Tensorflow..."
   openvino2tensorflow --load_dest_file_path_for_the_calib_npy $VBX_SDK/tutorials/imagenetv2_20x227x227x3.npy \
--model_path squeezenet1.1.xml \
--output_full_integer_quant_tflite \
--string_formulas_for_normalization '(data - [0.,0.,0.]) / [1.,1.,1.]'
   cp saved_model/model_full_integer_quant.tflite squeezenet1.1.tflite
fi

if [ -f squeezenet1.1.tflite ]; then
   tflite_preprocess squeezenet1.1.tflite   
fi

if [ -f squeezenet1.1.pre.tflite ]; then
    echo "Generating VNNX for V1000 ncomp configuration..."
    vnnx_compile -s V1000 -c ncomp -t squeezenet1.1.pre.tflite  -o squeezenet1.1_V1000_ncomp.vnnx
fi

if [ -f squeezenet1.1_V1000_ncomp.vnnx ]; then
    echo "Running Simulation..."
    python $VBX_SDK/example/python/classifier.py squeezenet1.1_V1000_ncomp.vnnx $VBX_SDK/tutorials/test_images/oreo.jpg 
    echo "C Simulation Command:"
    echo '$VBX_SDK/example/sim-c/sim-run-model squeezenet1.1_V1000_ncomp.vnnx $VBX_SDK/tutorials/test_images/oreo.jpg CLASSIFY'
fi

deactivate
