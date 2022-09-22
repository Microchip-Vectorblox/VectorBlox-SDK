
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

echo "Downloading yolov2-tiny-voc..."
[ -f yolov2-tiny-voc.cfg ] || wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov2-tiny-voc.cfg
[ -f yolov2-tiny-voc.weights ] || wget http://web.archive.org/web/20220320120309/https://pjreddie.com/media/files/yolov2-tiny-voc.weights
[ -f voc.names ] || wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/voc.names
python ../darknet_to_onnx.py yolov2-tiny-voc.cfg

echo "Running Model Optimizer..."
# model details @ https://pjreddie.com/darknet/yolo/
mo --input_model yolov2-tiny-voc.onnx \
--framework onnx \
--input_shape [1,3,416,416] \
--scale_values=[255.] \
--reverse_input_channels \
--static_shape

echo "Generating VNNX for V1000 configuration..."
generate_vnnx -x yolov2-tiny-voc.xml  -c V1000 -f ../../sample_images -o yolov2-tiny-voc.vnnx

echo "Running Simulation..."
python $VBX_SDK/example/python/yoloInfer.py yolov2-tiny-voc.vnnx ../../test_images/dog.jpg -j yolov2-tiny-voc.json -l voc.names

deactivate
