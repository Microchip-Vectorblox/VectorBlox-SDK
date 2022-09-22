
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

echo "Downloading retinaface.resnet..."
[ -f Resnet50_Final.pth ] || gdown 'http://drive.google.com/uc?id=14KX6VqF69MdSPk3Tr9PlDYbq7ArpdNUW&authuser=0&export=downloads'
[ -f Resnet50_Final.pth ] || exit 1
rm -rf Pytorch_Retinaface && git clone https://github.com/biubug6/Pytorch_Retinaface
cd Pytorch_Retinaface
sed -i 's/default=640/default=320/' convert_to_onnx.py #needed to fix bug
mkdir weights
cp ../Resnet50_Final.pth weights
python convert_to_onnx.py --trained_model ./weights/Resnet50_Final.pth --network resnet50
mv FaceDetector.onnx ../retinaface.resnet.onnx
cd ..

echo "Running Model Optimizer..."
# model details @ https://github.com/biubug6/Pytorch_Retinaface
mo --input_model retinaface.resnet.onnx \
--input_shape [1,3,320,320] \
--mean_values [104,117,123] \
--output=Conv_217,Conv_225,Conv_233,Conv_192,Conv_200,Conv_208,Conv_242,Conv_250,Conv_258 \
--static_shape

echo "Generating VNNX for V1000 configuration..."
generate_vnnx -x retinaface.resnet.xml  -c V1000 -f ../../sample_images -o retinaface.resnet.vnnx

echo "Running Simulation..."
python $VBX_SDK/example/python/retinaface.py retinaface.resnet.vnnx ../../test_images/3faces.jpg

deactivate
