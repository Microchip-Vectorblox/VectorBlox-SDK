
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

echo "Downloading yolov3-tiny..."
[ -f yolov3-tiny.cfg ] || wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg
[ -f yolov3-tiny.weights ] || wget https://pjreddie.com/media/files/yolov3-tiny.weights
[ -f coco.names ] || wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names
python ../darknet_to_onnx.py yolov3-tiny.cfg

echo "Running Model Optimizer..."
# model details @ https://pjreddie.com/darknet/yolo/
mo --input_model yolov3-tiny.onnx \
--framework onnx \
--input_shape [1,3,416,416] \
--scale_values=[255.] \
--reverse_input_channels \
--static_shape

echo "Generating VNNX for V1000 configuration..."
generate_vnnx -x yolov3-tiny.xml  -c V1000 -f ../../sample_images -o yolov3-tiny.vnnx

echo "Running Simulation..."
python $VBX_SDK/example/python/yoloInfer.py yolov3-tiny.vnnx ../../dog.jpg -j yolov3-tiny.json -l coco.names

deactivate
