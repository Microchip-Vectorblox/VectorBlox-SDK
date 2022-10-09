
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

echo "Downloading myfirstcnn..."
rm -rf myfirstcnn myfirstcnn.ipynb
[ -f myfirstcnn.zip ] || wget https://vector-blox-model-zoo.s3.us-west-2.amazonaws.com/Releases/ModelZoo/myfirstcnn.zip
unzip myfirstcnn.zip

echo "Running Model Optimizer..."
# model details @ 
mo --saved_model_dir ./myfirstcnn \
--input_shape [1,28,28,1] \
--scale_values [255] \
--static_shape

echo "Generating VNNX for V1000 configuration..."
generate_vnnx -x saved_model.xml  -c V1000 -f ../../sample_images -o myfirstcnn.vnnx

echo "Running Simulation..."
python $VBX_SDK/example/python/classifier.py myfirstcnn.vnnx ../../test_images/seven.28.jpg

deactivate
