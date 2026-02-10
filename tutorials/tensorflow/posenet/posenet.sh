
##########################################################
#  _    __          __             ____  __              #
# | |  / /__  _____/ /_____  _____/ __ )/ /___  _  __    #
# | | / / _ \/ ___/ __/ __ \/ ___/ __  / / __ \| |/_/    #
# | |/ /  __/ /__/ /_/ /_/ / /  / /_/ / / /_/ />  <      #
# |___/\___/\___/\__/\____/_/  /_____/_/\____/_/|_|      #
#                                                        #
# https://github.com/Microchip-Vectorblox/VectorBlox-SDK #
# v3.0                                                   #
#                                                        #
##########################################################

set -e
echo "Checking and activating VBX Python Environment..."
if [ -z $VBX_SDK ]; then
    echo "\$VBX_SDK not set. Please run 'source setup_vars.sh' from the SDK's root folder" && exit 1
fi
source $VBX_SDK/vbx_env/bin/activate

echo "Checking for Numpy calibration data file..."
if [ ! -f $VBX_SDK/tutorials/face_rgb_20x273x481x3.npy ]; then
    generate_npy $VBX_SDK/tutorials/face_rgb_20x273x481x3.npy -o $VBX_SDK/tutorials/face_rgb_20x273x481x3.npy -s 273 481 
fi

echo "Checking for posenet files..."

# model details @ https://github.com/tensorflow/tfjs-models/blob/master/posenet
if [ ! -f posenet.tflite ]; then
if [ ! -f mobilenet100_stride16.pb ]; then
# download model; the link for the .json file (and other posenet models) can be found here:
#   https://github.com/tensorflow/tfjs-models/blob/master/posenet/src/checkpoints.ts
wget -q https://storage.googleapis.com/tfjs-models/savedmodel/posenet/mobilenet/float/100/model-stride16.json -O mobilenet100_stride16.json
wget -q https://storage.googleapis.com/tfjs-models/savedmodel/posenet/mobilenet/float/100/group1-shard1of4.bin
wget -q https://storage.googleapis.com/tfjs-models/savedmodel/posenet/mobilenet/float/100/group1-shard2of4.bin
wget -q https://storage.googleapis.com/tfjs-models/savedmodel/posenet/mobilenet/float/100/group1-shard3of4.bin
wget -q https://storage.googleapis.com/tfjs-models/savedmodel/posenet/mobilenet/float/100/group1-shard4of4.bin

# this repo converts the tfjs graph to a tensorflow graph, which can then be imported into openvino
python3 -m venv tfjs_env
source tfjs_env/bin/activate
python -m pip install -U pip
python -m pip install tensorflow-estimator
python -m pip install tfjs-graph-converter
tfjs_graph_converter mobilenet100_stride16.json mobilenet100_stride16.pb
deactivate
source $VBX_SDK/vbx_env/bin/activate

fi
fi

if [ ! -d $VBX_SDK/example/python/posenet_python ]; then
# this repo is a python port of the javascript posenet; we will use its post-processing routine
   rm -rf $VBX_SDK/example/python/posenet_python
   rm -rf $VBX_SDK/example/python/posenet
   git clone https://github.com/rwightman/posenet-python $VBX_SDK/example/python/posenet_python
   cp -r $VBX_SDK/example/python/posenet_python/posenet $VBX_SDK/example/python/.

   mkdir -p output
fi


if [ ! -f posenet.tflite ]; then
   echo "Generating TF Lite..."
   tflite_quantize mobilenet100_stride16.pb posenet.tflite -d $VBX_SDK/tutorials/face_rgb_20x273x481x3.npy \
--mean 128 \
--scale 128 --shape 1 273 481 3
fi

if [ -f posenet.tflite ]; then
   tflite_preprocess posenet.tflite  --mean 128 --scale 128
fi

if [ -f posenet.pre.tflite ]; then
    echo "Generating VNNX for V1000 ncomp configuration..."
    vnnx_compile -s V1000 -c ncomp -t posenet.pre.tflite  -o posenet_V1000_ncomp.vnnx
fi

if [ -f posenet_V1000_ncomp.vnnx ]; then
    echo "Running Simulation..."
    python $VBX_SDK/example/python/posenetInfer.py posenet_V1000_ncomp.vnnx $VBX_SDK/tutorials/test_images/ski.273.481.jpg 
    echo "C Simulation Command:"
    echo '$VBX_SDK/example/sim-c/sim-run-model posenet_V1000_ncomp.vnnx $VBX_SDK/tutorials/test_images/ski.273.481.jpg  '
fi

deactivate
