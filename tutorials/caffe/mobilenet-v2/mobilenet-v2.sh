
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

echo "Downloading mobilenet-v2..."
omz_downloader --name mobilenet-v2

echo "Running Model Optimizer..."
# model details @ https://github.com/openvinotoolkit/open_model_zoo/tree/2021.4.2/models/public/mobilenet-v2
mo --input_model public/mobilenet-v2/mobilenet-v2.caffemodel \
--mean_values [103.94,116.78,123.68] \
--scale_values [58.82] \
--static_shape

echo "Generating VNNX for V1000 configuration..."
generate_vnnx -x mobilenet-v2.xml  -c V1000 -f ../../sample_images -o mobilenet-v2.vnnx --bias-correction

echo "Running Simulation..."
python $VBX_SDK/example/python/classifier.py mobilenet-v2.vnnx ../../test_images/oreo.jpg

deactivate
