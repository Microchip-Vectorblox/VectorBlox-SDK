
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

echo "Downloading onnx_squeezenet1.0..."
wget --no-check-certificate https://media.githubusercontent.com/media/onnx/models/main/vision/classification/squeezenet/model/squeezenet1.0-7.onnx

echo "Running Model Optimizer..."
# model details @ https://github.com/onnx/models/tree/main/vision/classification/squeezenet
mo --input_model squeezenet1.0-7.onnx \
--mean_values [123.675,116.28,103.53] \
--static_shape

echo "Generating VNNX for V1000 configuration..."
generate_vnnx -x squeezenet1.0-7.xml  -c V1000 -f ../../sample_images -o onnx_squeezenet1.0.vnnx --bias-correction

echo "Running Simulation..."
python $VBX_SDK/example/python/classifier.py onnx_squeezenet1.0.vnnx ../../test_images/oreo.jpg

deactivate
