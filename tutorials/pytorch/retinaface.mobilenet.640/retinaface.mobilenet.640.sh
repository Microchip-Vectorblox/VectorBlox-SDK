
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

echo "Downloading retinaface.mobilenet.640..."
[ -f mobilenet0.25_Final.pth ] || gdown 'https://drive.google.com/uc?id=15zP8BP-5IvWXWZoYTNdvUJUiBqZ1hxu1&authuser=0&export=download'
[ -f mobilenetV1X0.25_pretrain.tar ] || gdown 'https://drive.google.com/uc?id=1q36RaTZnpHVl4vRuNypoEMVWiiwCqhuD&authuser=0&export=download'
[ -f mobilenet0.25_Final.pth ] || exit 1
[ -f mobilenetV1X0.25_pretrain.tar ] || exit 1
rm -rf Pytorch_Retinaface && git clone https://github.com/biubug6/Pytorch_Retinaface
cd Pytorch_Retinaface
mkdir weights
cp ../mobilenet0.25_Final.pth weights
cp ../mobilenetV1X0.25_pretrain.tar weights
python convert_to_onnx.py --trained_model ./weights/mobilenet0.25_Final.pth --network mobile0.25
mv FaceDetector.onnx ../retinaface.mobilenet.640.onnx
cd ..

echo "Running Model Optimizer..."
# model details @ https://github.com/biubug6/Pytorch_Retinaface
mo --input_model retinaface.mobilenet.640.onnx \
--input_shape [1,3,640,640] \
--mean_values [104,117,123] \
--output=Conv_152,Conv_160,Conv_168,Conv_127,Conv_135,Conv_143,Conv_177,Conv_185,Conv_193 \
--static_shape

echo "Generating VNNX for V1000 configuration..."
generate_vnnx -x retinaface.mobilenet.640.xml  -c V1000 -f ../../sample_images -o retinaface.mobilenet.640.vnnx --bias-correction

echo "Running Simulation..."
python $VBX_SDK/example/python/retinaface.py retinaface.mobilenet.640.vnnx ../../test_images/3faces.jpg

deactivate
