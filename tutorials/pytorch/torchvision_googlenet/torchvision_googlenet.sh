
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

echo "Downloading torchvision_googlenet..."
python ../../torchvision_to_onnx.py googlenet

echo "Running Model Optimizer..."
# model details @ https://pytorch.org/docs/stable/torchvision/models.html
mo --input_model googlenet.onnx \
--reverse_input_channels \
--mean_values [123.675,116.28,103.53] \
--scale_values [58.395,57.12,57.375] \
--static_shape

echo "Generating VNNX for V1000 configuration..."
generate_vnnx -x googlenet.xml  -c V1000 -f ../../sample_images -o torchvision_googlenet.vnnx

echo "Running Simulation..."
python $VBX_SDK/example/python/classifier.py torchvision_googlenet.vnnx ../../oreo.jpg

deactivate
