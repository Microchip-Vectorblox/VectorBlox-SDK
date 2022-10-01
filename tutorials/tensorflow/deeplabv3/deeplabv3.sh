
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

echo "Downloading deeplabv3..."
omz_downloader --name deeplabv3

echo "Running Model Optimizer..."
# model details @ https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/deeplabv3
mo --input_model public/deeplabv3/deeplabv3_mnv2_pascal_train_aug/frozen_inference_graph.pb \
--input_shape=[1,513,513,3] \
--input=1:mul_1 \
--output=ArgMax \
--static_shape

echo "Generating VNNX for V1000 configuration..."
generate_vnnx -x frozen_inference_graph.xml  -c V1000 -f ../../sample_images -o deeplabv3.vnnx

echo "Running Simulation..."
python $VBX_SDK/example/python/deeplab.py deeplabv3.vnnx ../../test_images/dog.jpg

deactivate
