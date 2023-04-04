
#######################################
#                                     #
#  ____    ____  ______   ___   ___   #
#  \   \  /   / |   _  \  \  \ /  /   #
#   \   \/   /  |  |_)  |  \  V  /    #
#    \      /   |   _  <    >   <     #
#     \    /    |  |_)  |  /  ^  \    #
#      \__/     |______/  /__/ \__\   #
#                                     #
# Refer to Programmer's Guide         #
# for full details                    #
#                                     #
#                                     #
#######################################

set -e
echo "Checking and Activating VBX Python Environment..."
if [ -z $VBX_SDK ]; then
    echo "\$VBX_SDK not set. Please run 'source setup_vars.sh' from the SDK's root folder" && exit 1
fi
source $VBX_SDK/vbx_env/bin/activate

echo "Downloading genderage..."
[ -f genderage.onnx ] || gdown 'https://drive.google.com/uc?id=1Mm3TeUuaZOwmEMp0nGOddvgXCjpRodPU&authuser=0&export=download'
[ -f genderage.onnx ] || exit 1

echo "Running Model Optimizer..."
# model details @ https://github.com/deepinsight/insightface/tree/master/model_zoo#41-genderage
mo --input_model genderage.onnx \
--input_shape [1,3,96,96] \
--reverse_input_channels \
--mean_values [0,0,0] \
--scale_values [1] \
--static_shape \
--output fullyconnected0,fullyconnected1

echo "Generating VNNX for V1000 configuration..."
generate_vnnx -x genderage.xml  -c V1000 -f ../../sample_attributes -o genderage.vnnx --bias-correction --samples-count 48

echo "Running Simulation..."
python $VBX_SDK/example/python/genderage.py genderage.vnnx ../../test_images/John_faceAtr.jpg

deactivate
