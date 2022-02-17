
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

echo "Downloading mobilenet-v1-1.0-224-tf..."
omz_downloader --name mobilenet-v1-1.0-224-tf

echo "Running Model Optimizer..."
# model details @ https://github.com/opencv/open_model_zoo/blob/2021.3/models/public/mobilenet-v1-1.0-224-tf/mobilenet-v1-1.0-224-tf.md
mo --input_model public/mobilenet-v1-1.0-224-tf/mobilenet_v1_1.0_224_frozen.pb \
--input_shape [1,224,224,3] \
--reverse_input_channels \
--mean_values [127.5,127.5,127.5] \
--scale_values [127.5] \
--static_shape

echo "Generating VNNX for V1000 configuration..."
generate_vnnx -x mobilenet_v1_1.0_224_frozen.xml  -c V1000 -f ../../sample_images -o mobilenet-v1-1.0-224-tf.vnnx --bias-correction

echo "Running Simulation..."
python $VBX_SDK/example/python/classifier.py mobilenet-v1-1.0-224-tf.vnnx ../../oreo.jpg

deactivate
