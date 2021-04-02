
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

echo "Downloading mnist..."
wget https://media.githubusercontent.com/media/onnx/models/master/vision/classification/mnist/model/mnist-1.onnx -O mnist-1.onnx

echo "Running Model Optimizer..."
# model details @ https://github.com/onnx/models/tree/master/vision/classification/mnist
converter --input_model mnist-1.onnx

echo "Generating VNNX for V1000 configuration..."
generate_vnnx -x mnist-1.xml  -c V1000 -f ../../sample_images -o mnist.vnnx

echo "Running Simulation..."
python $VBX_SDK/example/python/mnist.py mnist.vnnx ../../seven.28.png

deactivate
