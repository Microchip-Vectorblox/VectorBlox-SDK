
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

echo "Downloading resnet-50-tf..."
wget -x https://download.01.org/opencv/public_models/012020/resnet-50-tf/resnet_v1-50.pb

echo "Running Model Optimizer..."
# model details @ https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/public/resnet-50-tf/resnet-50-tf.md
converter --input_model download.01.org/opencv/public_models/012020/resnet-50-tf/resnet_v1-50.pb \
--input_shape=[1,224,224,3] \
--mean_values=[123.68,116.78,103.94] \
--input=map/TensorArrayStack/TensorArrayGatherV3 \
--output=softmax_tensor \
--reverse_input_channels

echo "Generating VNNX for V1000 configuration..."
generate_vnnx -x resnet_v1-50.xml  -c V1000 -f ../../sample_images -o resnet-50-tf.vnnx

echo "Running Simulation..."
python $VBX_SDK/example/python/imagenet.py resnet-50-tf.vnnx ../../oreo.jpg

deactivate
