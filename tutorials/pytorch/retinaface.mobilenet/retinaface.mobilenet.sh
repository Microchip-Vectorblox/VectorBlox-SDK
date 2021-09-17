
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

echo "Downloading retinaface.mobilenet..."
rm -rf Pytorch_Retinaface && git clone https://github.com/biubug6/Pytorch_Retinaface
cd Pytorch_Retinaface
sed -i 's/args.long_side, args.long_side/288, 512/' convert_to_onnx.py # model input size 512x288; must be multiples of 32
python3 -m venv gdown
source gdown/bin/activate
pip install gdown
python gdown/bin/gdown 'https://drive.google.com/uc?id=15zP8BP-5IvWXWZoYTNdvUJUiBqZ1hxu1&authuser=0&export=download'
python gdown/bin/gdown 'https://drive.google.com/uc?id=1q36RaTZnpHVl4vRuNypoEMVWiiwCqhuD&authuser=0&export=download'
mkdir weights
mv mobilenet0.25_Final.pth weights
mv mobilenetV1X0.25_pretrain.tar weights
source $VBX_SDK/vbx_env/bin/activate
python convert_to_onnx.py --trained_model ./weights/mobilenet0.25_Final.pth --network mobile0.25
mv FaceDetector.onnx ../retinaface.mobilenet.onnx
cd ..

echo "Running Model Optimizer..."
# model details @ 
converter --input_model retinaface.mobilenet.onnx \
--input_shape [1,3,288,512] \
--mean_values [104,117,123] \
--output=Conv_199,Conv_207,Conv_215,Conv_174,Conv_182,Conv_190,Conv_224,Conv_232,Conv_240 \
--static_shape

echo "Generating VNNX for V1000 configuration..."
generate_vnnx -x retinaface.mobilenet.xml  -c V1000 -f ../../sample_images -o retinaface.mobilenet.vnnx --bias-correction

echo "Running Simulation..."
python $VBX_SDK/example/python/retinaface.py retinaface.mobilenet.vnnx ../../3faces.png  --width 512 --height 288

deactivate
