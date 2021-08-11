
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

echo "Downloading mobilenet-v1-1.0-224..."
downloader --name mobilenet-v1-1.0-224

echo "Running Model Optimizer..."
# model details @ https://github.com/opencv/open_model_zoo/blob/2019_R3.1/models/public/mobilenet-v1-1.0-224/mobilenet-v1-1.0-224.md
converter --input_model public/mobilenet-v1-1.0-224/mobilenet-v1-1.0-224.caffemodel \
--mean_values [103.94,116.78,123.68] \
--scale_values [58.82]

echo "Generating VNNX for V1000 configuration..."
generate_vnnx -x mobilenet-v1-1.0-224.xml  -c V1000 -f ../../sample_images -o mobilenet-v1-1.0-224.vnnx --bias-correction

echo "Running Simulation..."
python $VBX_SDK/example/python/imagenet.py mobilenet-v1-1.0-224.vnnx ../../oreo.jpg

deactivate
