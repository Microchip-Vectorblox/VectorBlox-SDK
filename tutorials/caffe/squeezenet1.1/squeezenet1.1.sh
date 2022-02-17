
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
omz_downloader --name squeezenet1.1

echo "Running Model Optimizer..."
# model details @ https://github.com/openvinotoolkit/open_model_zoo/blob/2021.3/models/public/squeezenet1.1/squeezenet1.1.md
mo --input_model public/squeezenet1.1/squeezenet1.1.caffemodel \
--mean_values [104,117,123] \
--input_shape [1,3,227,227] \
--static_shape

echo "Generating VNNX for V1000 configuration..."
generate_vnnx -x squeezenet1.1.xml  -c V1000 -f ../../sample_images -o squeezenet1.1.vnnx

echo "Running Simulation..."
python $VBX_SDK/example/python/classifier.py squeezenet1.1.vnnx ../../oreo.jpg --channels 3 --height 227 --width 227

deactivate
