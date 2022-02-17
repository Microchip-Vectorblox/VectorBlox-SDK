
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

echo "Downloading onnx_resnet101-v1..."
wget https://media.githubusercontent.com/media/onnx/models/main/vision/classification/resnet/model/resnet101-v1-7.onnx

echo "Running Model Optimizer..."
# model details @ https://github.com/onnx/models/tree/main/vision/classification/resnet
mo --input_model resnet101-v1-7.onnx \
--reverse_input_channels \
--mean_values [123.675,116.28,103.53] \
--scale_values [58.4,57.1,57.38] \
--static_shape

echo "Generating VNNX for V1000 configuration..."
generate_vnnx -x resnet101-v1-7.xml  -c V1000 -f ../../sample_images -o onnx_resnet101-v1.vnnx

echo "Running Simulation..."
python $VBX_SDK/example/python/classifier.py onnx_resnet101-v1.vnnx ../../oreo.jpg

deactivate
