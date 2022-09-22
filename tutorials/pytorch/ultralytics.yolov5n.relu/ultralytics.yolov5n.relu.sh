
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

echo "Downloading ultralytics.yolov5n.relu..."
[ -f coco.names ] || wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names
wget https://vector-blox-model-zoo.s3.us-west-2.amazonaws.com/Releases/ModelZoo/ultralytics.yolov5n.relu.onnx

echo "Running Model Optimizer..."
# model details @ https://github.com/ultralytics/yolov5
mo --input_model ultralytics.yolov5n.relu.onnx \
--framework onnx \
--input_shape [1,3,416,416] \
--scale_values=[255.] \
--reverse_input_channels \
--output=Conv_193,Conv_247,Conv_139 \
--static_shape

echo "Generating VNNX for V1000 configuration..."
generate_vnnx -x ultralytics.yolov5n.relu.xml  -c V1000 -f ../../sample_images -o ultralytics.yolov5n.relu.vnnx --bias-correction

echo "Running Simulation..."
python $VBX_SDK/example/python/yoloInfer.py ultralytics.yolov5n.relu.vnnx ../../test_images/dog.jpg -j yolov5n.json -v 5 -l coco.names -t 0.25

deactivate
