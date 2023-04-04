
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

echo "Downloading inception_v3..."
python - <<EOF
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3
inception_v3_model = InceptionV3(input_shape=(299, 299, 3), weights='imagenet', classifier_activation=None)
inception_v3_model.save('saved_model/')
EOF

echo "Running Model Optimizer..."
# model details @ https://keras.io/api/applications/inceptionv3/
mo --saved_model_dir saved_model \
--input_shape=[1,299,299,3] \
--reverse_input_channels \
--mean_values [127.5,127.5,127.5] \
--scale_values [127.5] \
--static_shape

echo "Generating VNNX for V1000 configuration..."
generate_vnnx -x saved_model.xml  -c V1000 -f ../../sample_images -o inception_v3.vnnx

echo "Running Simulation..."
python $VBX_SDK/example/python/classifier.py inception_v3.vnnx ../../test_images/oreo.jpg

deactivate
