
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

echo "Downloading scrfd_500m_bnkps..."
# ONNX file was created through the following process:
# git clone https://github.com/deepinsight/insightface.git
# install https://github.com/deepinsight/insightface/tree/master/detection/scrfd#installation
# download SCRFD_500M_KPS https://github.com/deepinsight/insightface/tree/master/detection/scrfd#pretrained-models
# python tools/scrfd2onnx.py configs/scrfd/scrfd_500m_bnkps.py scrfd_500m_bnkps.pth --input-img garden_512x288.jpg --output-file scrfd_500m_bnkps.onnx

[ -f scrfd_500m_bnkps.onnx ] || wget https://vector-blox-model-zoo.s3.us-west-2.amazonaws.com/Releases/ModelZoo/scrfd_500m_bnkps.onnx

echo "Running Model Optimizer..."
# model details @ https://insightface.ai/scrfd
mo --input_model scrfd_500m_bnkps.onnx \
--input_shape [1,3,288,512] \
--reverse_input_channels \
--mean_values [127.5,127.5,127.5] \
--scale_values [128] \
--static_shape \
--output=Conv_160,Conv_161,Conv_162,Conv_139,Conv_140,Conv_141,Conv_118,Conv_119,Conv_120

echo "Generating VNNX for V1000 configuration..."
generate_vnnx -x scrfd_500m_bnkps.xml  -c V1000 -f ../../sample_images -o scrfd_500m_bnkps.vnnx --bias-correction

echo "Running Simulation..."
python $VBX_SDK/example/python/scrfdInfer.py scrfd_500m_bnkps.vnnx ../../test_images/garden.jpg

deactivate
