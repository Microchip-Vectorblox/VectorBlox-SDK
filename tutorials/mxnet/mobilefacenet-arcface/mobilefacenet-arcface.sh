
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

echo "Downloading mobilefacenet-arcface..."
[ -f model-y1.zip ] || gdown 'https://drive.google.com/uc?id=1RHyJIeYuHduVDDBTn3ffpYEZoXWRamWI&authuser=0&export=download'
[ -f model-y1.zip ] || exit 1
rm -rf arcface && mkdir arcface
cd arcface
cp ../model-y1.zip .
unzip model-y1.zip
cd ..

echo "Running Model Optimizer..."
# model details @ https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/public/face-recognition-mobilefacenet-arcface/model.yml
mo --input_model=arcface/model-y1-test2/model-0000.params \
--reverse_input_channels \
--input_shape=[1,3,112,112] \
--static_shape

echo "Generating VNNX for V1000 configuration..."
generate_vnnx -x model-0000.xml  -c V1000 -f ../../sample_images -o mobilefacenet-arcface.vnnx --bias-correction

echo "Running Simulation..."
python $VBX_SDK/example/python/face_compare.py mobilefacenet-arcface.vnnx ../../MattDamon0001_arcface.png ../../MattDamon0002_arcface.png --height 112 --width 112

deactivate
