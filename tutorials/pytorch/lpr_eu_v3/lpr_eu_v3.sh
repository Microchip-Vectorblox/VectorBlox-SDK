
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

echo "Downloading lpr_eu_v3..."
[ -f lpr_eu_v3.onnx ] || wget https://vector-blox-model-zoo.s3.us-west-2.amazonaws.com/Releases/ModelZoo/lpr_eu_v3.onnx

echo "Running Model Optimizer..."
# model details @ 
mo --input_model lpr_eu_v3.onnx \
--scale_values [255.] \
--reverse_input_channels \
--static_shape

echo "Generating VNNX for V1000 configuration..."
generate_vnnx -x lpr_eu_v3.xml  -c V1000 -f ../../sample_plates  -o lpr_eu_v3.vnnx --bias-correction 

echo "Running Simulation..."
python $VBX_SDK/example/python/recognize_plate_eu.py lpr_eu_v3.vnnx ../../test_images/A358CC82.jpg

deactivate
