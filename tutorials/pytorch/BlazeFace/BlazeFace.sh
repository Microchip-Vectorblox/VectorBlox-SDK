
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

echo "Downloading BlazeFace..."
rm -rf BlazeFace-PyTorch
git clone https://github.com/hollance/BlazeFace-PyTorch
cd BlazeFace-PyTorch

python -c "import torch.onnx;
from blazeface import BlazeFace; \
PATH='blazeface.pth';\
model = BlazeFace(); \
model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')));\
model.eval();\
x = torch.randn(1,3,128,128);\
torch.onnx.export(model, x, 'blazeface.onnx');"

cd ..
cp BlazeFace-PyTorch/blazeface.onnx .

echo "Running Model Optimizer..."
# model details @ https://github.com/hollance/BlazeFace-PyTorch
mo --input_model blazeface.onnx \
--reverse_input_channels \
--mean_values [127.5,127.5,127.5] \
--scale_values [127.5] \
--static_shape

echo "Generating VNNX for V1000 configuration..."
generate_vnnx -x blazeface.xml  -c V1000 -f ../../sample_images -o BlazeFace.vnnx --bias-correction

echo "Running Simulation..."
python $VBX_SDK/example/python/blazeface.py -a BlazeFace-PyTorch/anchors.npy BlazeFace.vnnx ../../test_images/3faces.jpg

deactivate
