
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

echo "Downloading torchvision_inception_v3..."
python ../../torchvision_to_onnx.py inception_v3

echo "Running Model Optimizer..."
# model details @ https://pytorch.org/vision/0.9/models.html#torchvision.models.inception_v3
mo --input_model inception_v3.onnx \
--reverse_input_channels \
--mean_values [123.675,116.28,103.53] \
--scale_values [58.395,57.12,57.375] \
--static_shape

echo "Generating VNNX for V1000 configuration..."
generate_vnnx -x inception_v3.xml  -c V1000 -f ../../sample_images -o torchvision_inception_v3.vnnx

echo "Running Simulation..."
python $VBX_SDK/example/python/classifier.py torchvision_inception_v3.vnnx ../../test_images/oreo.jpg

deactivate
