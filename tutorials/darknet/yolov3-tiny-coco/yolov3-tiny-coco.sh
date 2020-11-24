
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

echo "Downloading yolov3-tiny-coco..."
rm -rf tensorflow-yolo-v3
git clone https://github.com/mystic123/tensorflow-yolo-v3

[ -f yolov3-tiny.weights ] || wget https://pjreddie.com/media/files/yolov3-tiny.weights
[ -f coco.names ] || wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names
cd tensorflow-yolo-v3
python ./convert_weights_pb.py --tiny --class_names ../coco.names --weights_file ../yolov3-tiny.weights --data_format NHWC

cd ../
cp tensorflow-yolo-v3/frozen_darknet_yolov3_model.pb yolov3-tiny.pb

echo "Running Model Optimizer..."
# model details @ https://pjreddie.com/darknet/yolo/
converter --input_model yolov3-tiny.pb \
--framework tf \
--input_shape [1,416,416,3] \
--reverse_input_channels \
--tensorflow_use_custom_operations_config $VBX_SDK/python/third_party/dldt/model-optimizer/extensions/front/tf/yolo_v3_tiny.json \
--static_shape 

echo "Generating VNNX for V1000 configuration..."
generate_vnnx -x yolov3-tiny.xml  -c V1000 -f ../../sample_images -o yolov3-tiny-coco.vnnx

echo "Running Simulation..."
python $VBX_SDK/example/python/yolov3.py yolov3-tiny-coco.vnnx ../../dog.416.jpg

deactivate
