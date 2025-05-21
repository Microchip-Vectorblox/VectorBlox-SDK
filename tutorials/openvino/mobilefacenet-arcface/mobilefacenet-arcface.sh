
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

echo "Downloading mobilefacenet-arcface..."
# model details @ https://github.com/deepinsight/insightface
wget -q --no-check-certificate https://github.com/Microchip-Vectorblox/assets/releases/download/assets/model-0000.xml
wget -q --no-check-certificate https://github.com/Microchip-Vectorblox/assets/releases/download/assets/model-0000.bin
mv model-0000.xml mobilefacenet-arcface.xml
mv model-0000.bin mobilefacenet-arcface.bin

echo "Running OpenVINO2Tensorflow..."
openvino2tensorflow --load_dest_file_path_for_the_calib_npy $VBX_SDK/tutorials/imagenetv2_20x224x224x3.npy \
--model_path mobilefacenet-arcface.xml \
--output_full_integer_quant_tflite \
--string_formulas_for_normalization '(data - [0.,0.,0.]) / [1.,1.,1.]'
cp saved_model/model_full_integer_quant.tflite mobilefacenet-arcface.tflite

if [ -f mobilefacenet-arcface.tflite ]; then
   tflite_preprocess mobilefacenet-arcface.tflite   
fi

if [ -f mobilefacenet-arcface.pre.tflite ]; then
    echo "Generating VNNX for V1000 configuration..."
    vnnx_compile -c V1000 -t mobilefacenet-arcface.pre.tflite -o mobilefacenet-arcface.vnnx
fi

if [ -f mobilefacenet-arcface.vnnx ]; then
    echo "Running Simulation..."
    python $VBX_SDK/example/python/face_compare.py mobilefacenet-arcface.vnnx $VBX_SDK/tutorials/test_images/MattDamon0001_arcface.jpg $VBX_SDK/tutorials/test_images/MattDamon0002_arcface.jpg 
    echo "C Simulation Command:"
    echo '$VBX_SDK/example/sim-c/sim-run-model mobilefacenet-arcface.vnnx $VBX_SDK/tutorials/test_images/MattDamon0001_arcface.jpg  '
fi

deactivate
