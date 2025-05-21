
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
if [ ! -f $VBX_SDK/tutorials/imagenetv2_20x224x224x3.npy ]; then
    generate_npy $VBX_SDK/tutorials/imagenetv2_rgb_20x224x224x3.npy -o $VBX_SDK/tutorials/imagenetv2_20x224x224x3.npy -s 224 224  -b 
fi

echo "Downloading mobilenet-v2-1.0-224..."
# model details @ https://github.com/openvinotoolkit/open_model_zoo/tree/2021.4.2/models/public/mobilenet-v2-1.0-224
omz_downloader --name mobilenet-v2-1.0-224

echo "Running Model Optimizer..."
mo --input_model public/mobilenet-v2-1.0-224/mobilenet_v2_1.0_224_frozen.pb \
--input_shape [1,224,224,3] \
--mean_values [127.5,127.5,127.5] \
--scale_values [127.5] \
--static_shape

echo "Running OpenVINO2Tensorflow..."
openvino2tensorflow --load_dest_file_path_for_the_calib_npy $VBX_SDK/tutorials/imagenetv2_20x224x224x3.npy \
--keep_input_tensor_in_nchw \
--weight_replacement_config fix.json \
--model_path mobilenet_v2_1.0_224_frozen.xml \
--output_full_integer_quant_tflite \
--string_formulas_for_normalization '(data - [0.,0.,0.]) / [1.,1.,1.]'
cp saved_model/model_full_integer_quant.tflite mobilenet-v2-1.0-224.tflite

if [ -f mobilenet-v2-1.0-224.tflite ]; then
   tflite_preprocess mobilenet-v2-1.0-224.tflite   
fi

if [ -f mobilenet-v2-1.0-224.pre.tflite ]; then
    echo "Generating VNNX for V1000 configuration..."
    vnnx_compile -c V1000 -t mobilenet-v2-1.0-224.pre.tflite -o mobilenet-v2-1.0-224.vnnx
fi

if [ -f mobilenet-v2-1.0-224.vnnx ]; then
    echo "Running Simulation..."
    python $VBX_SDK/example/python/classifier.py mobilenet-v2-1.0-224.vnnx $VBX_SDK/tutorials/test_images/oreo.jpg 
    echo "C Simulation Command:"
    echo '$VBX_SDK/example/sim-c/sim-run-model mobilenet-v2-1.0-224.vnnx $VBX_SDK/tutorials/test_images/oreo.jpg CLASSIFY'
fi

deactivate
