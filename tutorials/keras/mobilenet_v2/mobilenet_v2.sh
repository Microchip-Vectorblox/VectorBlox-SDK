
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

echo "Downloading mobilenet_v2..."
python - <<EOF
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
mobilenet_v2_model = MobileNetV2(input_shape=(224, 224, 3), weights='imagenet', classifier_activation=None)
mobilenet_v2_model.save('saved_model/')
EOF

echo "Running Model Optimizer..."
# model details @ https://keras.io/api/applications/mobilenet/
mo --saved_model_dir saved_model \
--input_shape=[1,224,224,3] \
--reverse_input_channels \
--mean_values [127.5,127.5,127.5] \
--scale_values [127.5] \
--static_shape

echo "Generating VNNX for V1000 configuration..."
generate_vnnx -x saved_model.xml  -c V1000 -f ../../sample_images -o mobilenet_v2.vnnx

echo "Running Simulation..."
python $VBX_SDK/example/python/classifier.py mobilenet_v2.vnnx ../../test_images/oreo.jpg

deactivate
