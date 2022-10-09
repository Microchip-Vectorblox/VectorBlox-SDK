
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

echo "Downloading posenet..."
# download model; the link for the .json file (and other posenet models) can be found here:
#   https://github.com/tensorflow/tfjs-models/blob/master/posenet/src/checkpoints.ts
wget https://storage.googleapis.com/tfjs-models/savedmodel/posenet/mobilenet/float/100/model-stride16.json -O mobilenet100_stride16.json
wget https://storage.googleapis.com/tfjs-models/savedmodel/posenet/mobilenet/float/100/group1-shard1of4.bin
wget https://storage.googleapis.com/tfjs-models/savedmodel/posenet/mobilenet/float/100/group1-shard2of4.bin
wget https://storage.googleapis.com/tfjs-models/savedmodel/posenet/mobilenet/float/100/group1-shard3of4.bin
wget https://storage.googleapis.com/tfjs-models/savedmodel/posenet/mobilenet/float/100/group1-shard4of4.bin

# this repo converts the tfjs graph to a tensorflow graph, which can then be imported into openvino
python3 -m venv tfjs_env
source tfjs_env/bin/activate
python -m pip install -U pip
python -m pip install tensorflow-estimator
python -m pip install tfjs-graph-converter
tfjs_graph_converter mobilenet100_stride16.json mobilenet100_stride16.pb
deactivate
source $VBX_SDK/vbx_env/bin/activate

# this repo is a python port of the javascript posenet; we will use its post-processing routine
rm -rf $VBX_SDK/example/python/posenet_python
rm -rf $VBX_SDK/example/python/posenet
git clone https://github.com/rwightman/posenet-python $VBX_SDK/example/python/posenet_python
cp -r $VBX_SDK/example/python/posenet_python/posenet $VBX_SDK/example/python/.

mkdir -p output

echo "Running Model Optimizer..."
# model details @ https://github.com/tensorflow/tfjs-models/blob/master/posenet
mo --input_model mobilenet100_stride16.pb \
--framework tf \
--input_shape [1,273,481,3] \
--mean_values [128.0,128.0,128.0] \
--scale_values [128.0] \
--static_shape

echo "Generating VNNX for V1000 configuration..."
generate_vnnx -x mobilenet100_stride16.xml  -c V1000 -f ../../sample_images -o posenet.vnnx --bias-correction

echo "Running Simulation..."
python $VBX_SDK/example/python/posenetInfer.py posenet.vnnx -i ../../test_images/ski.273.481.jpg

deactivate
