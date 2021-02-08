
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

echo "Downloading yolov2..."
[ -f yolov2.cfg ] || wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov2.cfg
[ -f yolov2.weights ] || wget https://pjreddie.com/media/files/yolov2.weights
[ -f coco.names ] || wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names
python $VBX_SDK/example/python/darknet_to_onnx.py yolov2.cfg

echo "Running Model Optimizer..."
# model details @ https://pjreddie.com/darknet/yolo/
converter --input_model yolov2.onnx \
--framework onnx \
--input_shape [1,3,608,608] \
--scale_values=[255.] \
--static_shape 

echo "Generating VNNX for V1000 configuration..."
generate_vnnx -x yolov2.xml  -c V1000 -f ../../sample_images -o yolov2.vnnx

echo "Running Simulation..."
python $VBX_SDK/example/python/yoloInfer.py yolov2.vnnx ../../dog.jpg -j yolov2.json -l coco.names

deactivate
