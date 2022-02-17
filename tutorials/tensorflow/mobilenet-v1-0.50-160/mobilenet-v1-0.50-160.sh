
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

echo "Downloading mobilenet-v1-0.50-160..."
omz_downloader --name mobilenet-v1-0.50-160

echo "Running Model Optimizer..."
# model details @ https://github.com/opencv/open_model_zoo/blob/2021.3/models/public/mobilenet-v1-0.50-160/mobilenet-v1-0.50-160.md
mo --input_model public/mobilenet-v1-0.50-160/mobilenet_v1_0.5_160_frozen.pb \
--input_shape [1,160,160,3] \
--reverse_input_channels \
--mean_values [127.5,127.5,127.5] \
--scale_values [127.5] \
--static_shape

echo "Generating VNNX for V1000 configuration..."
generate_vnnx -x mobilenet_v1_0.5_160_frozen.xml  -c V1000 -f ../../sample_images -o mobilenet-v1-0.50-160.vnnx --bias-correction

echo "Running Simulation..."
python $VBX_SDK/example/python/classifier.py mobilenet-v1-0.50-160.vnnx ../../oreo.jpg --channels 3 --height 160 --width 160

deactivate
