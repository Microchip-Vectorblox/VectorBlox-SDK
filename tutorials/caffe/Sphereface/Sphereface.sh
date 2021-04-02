
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

echo "Downloading Sphereface..."
downloader --name Sphereface

echo "Running Model Optimizer..."
# model details @ https://github.com/openvinotoolkit/open_model_zoo/blob/2020.4/models/public/Sphereface/Sphereface.md
converter --input_model public/Sphereface/Sphereface.caffemodel \
--input_shape=[1,3,112,96] \
--mean_values [127.5,127.5,127.5] \
--scale_values [128.0] \
--static_shape

echo "Generating VNNX for V1000 configuration..."
generate_vnnx -x Sphereface.xml  -c V1000 -f ../../sample_images -o Sphereface.vnnx

echo "Running Simulation..."
python $VBX_SDK/example/python/face_compare.py Sphereface.vnnx ../../matt-damon1_aligned.png ../../matt-damon2_aligned.png

deactivate
