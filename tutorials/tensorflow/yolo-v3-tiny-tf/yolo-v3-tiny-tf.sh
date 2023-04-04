
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

echo "Downloading yolo-v3-tiny-tf..."
[ -f coco.names ] || wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names
[ -f yolov3-tiny.weights ] || wget http://web.archive.org/web/20210225040312/https://pjreddie.com/media/files/yolov3-tiny.weights
omz_downloader --name yolo-v3-tiny-tf
rm -rf keras-YOLOv3-model-set && git clone https://github.com/david8862/keras-YOLOv3-model-set
cd keras-YOLOv3-model-set && git checkout 56bcc2e && cd ..
python keras-YOLOv3-model-set/tools/model_converter/convert.py keras-YOLOv3-model-set/cfg/yolov3-tiny.cfg yolov3-tiny.weights yolo-v3-tiny.h5
python -m pip install keras_applications
python keras-YOLOv3-model-set/tools/model_converter/keras_to_tensorflow.py --input_model=yolo-v3-tiny.h5 --output_model=yolo-v3-tiny.pb

echo "Running Model Optimizer..."
# model details @ https://github.com/openvinotoolkit/open_model_zoo/tree/2021.4.2/models/public/yolo-v3-tf/
mo --input_model=yolo-v3-tiny.pb \
--framework tf \
--input_shape [1,416,416,3] \
--scale_values [255.] \
--reverse_input_channels \
--static_shape

echo "Generating VNNX for V1000 configuration..."
generate_vnnx -x yolo-v3-tiny.xml  -c V1000 -f ../../sample_images -o yolo-v3-tiny-tf.vnnx --bias-correction

echo "Running Simulation..."
python $VBX_SDK/example/python/yoloInfer.py yolo-v3-tiny-tf.vnnx ../../test_images/dog.jpg -j yolo-v3-tiny-tf.json -l coco.names

deactivate
