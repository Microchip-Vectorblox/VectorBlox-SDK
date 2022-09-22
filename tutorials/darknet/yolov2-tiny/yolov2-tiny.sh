
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

echo "Downloading yolov2-tiny..."
[ -f yolov2-tiny.cfg ] || wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov2-tiny.cfg
[ -f yolov2-tiny.weights ] || wget http://web.archive.org/web/20220325031036/https://pjreddie.com/media/files/yolov2-tiny.weights
[ -f coco.names ] || wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names
python ../darknet_to_onnx.py yolov2-tiny.cfg

echo "Running Model Optimizer..."
# model details @ https://pjreddie.com/darknet/yolo/
mo --input_model yolov2-tiny.onnx \
--framework onnx \
--input_shape [1,3,416,416] \
--scale_values=[255.] \
--reverse_input_channels \
--static_shape

echo "Generating VNNX for V1000 configuration..."
generate_vnnx -x yolov2-tiny.xml  -c V1000 -f ../../sample_images -o yolov2-tiny.vnnx

echo "Running Simulation..."
python $VBX_SDK/example/python/yoloInfer.py yolov2-tiny.vnnx ../../test_images/dog.jpg -j yolov2-tiny.json -l coco.names

deactivate
