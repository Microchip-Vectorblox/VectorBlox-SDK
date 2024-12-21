
##########################################################
#  _    __          __             ____  __              #
# | |  / /__  _____/ /_____  _____/ __ )/ /___  _  __    #
# | | / / _ \/ ___/ __/ __ \/ ___/ __  / / __ \| |/_/    #
# | |/ /  __/ /__/ /_/ /_/ / /  / /_/ / / /_/ />  <      #
# |___/\___/\___/\__/\____/_/  /_____/_/\____/_/|_|      #
#                                                        #
# https://github.com/Microchip-Vectorblox/VectorBlox-SDK #
# v2.0                                                   #
#                                                        #
##########################################################

set -e
echo "Checking and activating VBX Python Environment..."
if [ -z $VBX_SDK ]; then
    echo "\$VBX_SDK not set. Please run 'source setup_vars.sh' from the SDK's root folder" && exit 1
fi
source $VBX_SDK/vbx_env/bin/activate

echo "Checking for Numpy calibration data file..."
if [ ! -f $VBX_SDK/tutorials/coco2017_rgb_20x416x416x3.npy ]; then
    wget -P $VBX_SDK/tutorials/ https://vector-blox-model-zoo.s3.us-west-2.amazonaws.com/EAP/calib_npy/coco2017_rgb_20x416x416x3.npy
fi

echo "Downloading yolo-v4-tiny-tf..."
# model details @ https://github.com/openvinotoolkit/open_model_zoo/tree/2021.4.2/models/public/yolo-v4-tiny-tf/
[ -f coco.names ] || wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names
omz_downloader --name yolo-v4-tiny-tf
DOWNLOAD_DIR=public/yolo-v4-tiny-tf
python $DOWNLOAD_DIR/keras-YOLOv3-model-set/tools/model_converter/convert.py $DOWNLOAD_DIR/keras-YOLOv3-model-set/cfg/yolov4-tiny.cfg $DOWNLOAD_DIR/yolov4-tiny.weights yolo-v4-tiny.h5

echo "Generating TF Lite..."
tflite_quantize yolo-v4-tiny.h5 yolo-v4-tiny-tf.tflite -d $VBX_SDK/tutorials/coco2017_rgb_20x416x416x3.npy \
--scale 255. --shape 1 416 416 3

if [ -f yolo-v4-tiny-tf.tflite ]; then
   tflite_preprocess yolo-v4-tiny-tf.tflite  --scale 255
fi

if [ -f yolo-v4-tiny-tf.pre.tflite ]; then
    echo "Generating VNNX for V1000 configuration..."
    vnnx_compile -c V1000 -t yolo-v4-tiny-tf.pre.tflite -o yolo-v4-tiny-tf.vnnx
fi

if [ -f yolo-v4-tiny-tf.vnnx ]; then
    echo "Running Simulation..."
    python $VBX_SDK/example/python/yoloInfer.py yolo-v4-tiny-tf.vnnx $VBX_SDK/tutorials/test_images/dog.jpg -j yolo-v4-tiny-tf.json -l coco.names -i 0.3 
fi

deactivate
