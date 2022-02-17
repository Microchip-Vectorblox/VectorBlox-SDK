
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

echo "Downloading license-plate-recognition-barrier-0007..."
omz_downloader --name license-plate-recognition-barrier-0007

echo "Running Model Optimizer..."
# model details @ https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/public/
mo --input_model=public/license-plate-recognition-barrier-0007/license-plate-recognition-barrier-0007/graph.pb.frozen \
--framework tf \
--input_shape [1,24,94,3] \
--reverse_input_channels \
--scale_values=[255.0] \
--output=Conv_15/BiasAdd \
--static_shape

echo "Generating VNNX for V1000 configuration..."
generate_vnnx -x graph.pb.xml  -c V1000 -f ../../sample_images -o license-plate-recognition-barrier-0007.vnnx --bias-correction

echo "Running Simulation..."
python $VBX_SDK/example/python/plate_recognition.py license-plate-recognition-barrier-0007.vnnx ../../Sichuan.jpg -c chinese

deactivate
