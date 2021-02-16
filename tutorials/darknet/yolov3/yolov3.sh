
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

echo "Downloading yolov3..."
[ -f yolov3.cfg ] || wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg
[ -f yolov3.weights ] || wget https://pjreddie.com/media/files/yolov3.weights
[ -f coco.names ] || wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names
python $VBX_SDK/example/python/darknet_to_onnx.py yolov3.cfg

echo "Running Model Optimizer..."
# model details @ https://pjreddie.com/darknet/yolo/
converter --input_model yolov3.onnx \
--framework onnx \
--input_shape [1,3,608,608] \
--scale_values=[255.] \
--reverse_input_channels \
--static_shape 

echo "Generating VNNX for V1000 configuration..."
generate_vnnx -x yolov3.xml  -c V1000 -f ../../sample_images -o yolov3.vnnx

echo "Running Simulation..."
python $VBX_SDK/example/python/yoloInfer.py yolov3.vnnx ../../dog.jpg -j yolov3.json -l coco.names

deactivate
