
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
    wget -P $VBX_SDK/tutorials/ https://vector-blox-model-zoo.s3.us-west-2.amazonaws.com/EAP/calib_npy/imagenetv2_20x224x224x3.npy
fi

echo "Downloading mobilenet-v1-1.0-224-tf..."
# model details @ https://github.com/openvinotoolkit/open_model_zoo/tree/2021.4.2/models/public/mobilenet-v1-1.0-224-tf/
omz_downloader --name mobilenet-v1-1.0-224-tf

echo "Running Model Optimizer..."
mo --input_model public/mobilenet-v1-1.0-224-tf/mobilenet_v1_1.0_224_frozen.pb \
--input_shape [1,224,224,3] \
--mean_values [127.5,127.5,127.5] \
--scale_values [127.5] \
--static_shape

echo "Running OpenVINO2Tensorflow..."
openvino2tensorflow --load_dest_file_path_for_the_calib_npy $VBX_SDK/tutorials/imagenetv2_20x224x224x3.npy \
--keep_input_tensor_in_nchw \
--weight_replacement_config fix.json \
--model_path mobilenet_v1_1.0_224_frozen.xml \
--output_full_integer_quant_tflite \
--string_formulas_for_normalization '(data - [0.,0.,0.]) / [1.,1.,1.]'
cp saved_model/model_full_integer_quant.tflite mobilenet-v1-1.0-224-tf.tflite

if [ -f mobilenet-v1-1.0-224-tf.tflite ]; then
   tflite_preprocess mobilenet-v1-1.0-224-tf.tflite   
fi

if [ -f mobilenet-v1-1.0-224-tf.pre.tflite ]; then
    echo "Generating VNNX for V1000 configuration..."
    vnnx_compile -c V1000 -t mobilenet-v1-1.0-224-tf.pre.tflite -o mobilenet-v1-1.0-224-tf.vnnx
fi

if [ -f mobilenet-v1-1.0-224-tf.vnnx ]; then
    echo "Running Simulation..."
    python $VBX_SDK/example/python/classifier.py mobilenet-v1-1.0-224-tf.vnnx $VBX_SDK/tutorials/test_images/oreo.jpg 
fi

deactivate