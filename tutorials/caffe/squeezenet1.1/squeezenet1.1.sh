
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

echo "Downloading squeezenet1.1..."
downloader --name squeezenet1.1

echo "Running Model Optimizer..."
# model details @ https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/public/squeezenet1.1/squeezenet1.1.md
converter --input_model public/squeezenet1.1/squeezenet1.1.caffemodel \
--mean_values [104,117,123]

echo "Generating VNNX for V1000 configuration..."
generate_vnnx -x squeezenet1.1.xml  -c V1000 -f ../../sample_images -o squeezenet1.1.vnnx

echo "Running Simulation..."
python $VBX_SDK/example/python/imagenet.py squeezenet1.1.vnnx ../../oreo.227.jpg

deactivate
