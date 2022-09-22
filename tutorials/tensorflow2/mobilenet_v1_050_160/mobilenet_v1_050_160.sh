
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

echo "Downloading mobilenet_v1_050_160..."
wget https://tfhub.dev/google/imagenet/mobilenet_v1_050_160/classification/5?tf-hub-format=compressed -O mobilenet_v1_050_160.tar.gz
mkdir -p mobilenet_v1_050_160
tar -xzf mobilenet_v1_050_160.tar.gz -C mobilenet_v1_050_160
python ../../saved_model_signature.py mobilenet_v1_050_160

echo "Running Model Optimizer..."
# model details @ https://tfhub.dev/google/imagenet/mobilenet_v1_050_160/classification/5
mo --saved_model_dir mobilenet_v1_050_160 \
--input_shape [1,160,160,3] \
--scale_values [255.0] \
--reverse_input_channels \
--static_shape

echo "Generating VNNX for V1000 configuration..."
generate_vnnx -x saved_model.xml  -c V1000 -f ../../sample_images -o mobilenet_v1_050_160.vnnx --bias-correction

echo "Running Simulation..."
python $VBX_SDK/example/python/classifier.py mobilenet_v1_050_160.vnnx ../../test_images/oreo.jpg

deactivate
