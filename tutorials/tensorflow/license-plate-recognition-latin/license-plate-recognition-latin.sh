
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

echo "Downloading license-plate-recognition-latin..."
rm -rf license-plate-recognition-latin
wget https://vector-blox-model-zoo.s3.us-west-2.amazonaws.com/Releases/ModelZoo/license-plate-recognition-latin.zip
unzip license-plate-recognition-latin.zip

echo "Running Model Optimizer..."
# model details @ https://github.com/openvinotoolkit/training_extensions/tree/misc/misc/tensorflow_toolkit/lpr
mo --input_model=license-plate-recognition-latin/graph.pb.frozen \
--framework tf \
--input_shape [1,24,112,3] \
--reverse_input_channels \
--scale_values=[255.0] \
--output=Conv_15/BiasAdd \
--static_shape

echo "Generating VNNX for V1000 configuration..."
generate_vnnx -x graph.pb.xml  -c V1000 -f ../../sample_images -o license-plate-recognition-latin.vnnx --bias-correction

echo "Running Simulation..."
python $VBX_SDK/example/python/plate_recognition.py license-plate-recognition-latin.vnnx ../../test_images/A358CC82.jpg -c latin

deactivate
