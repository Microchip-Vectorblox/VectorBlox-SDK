
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
    generate_npy $VBX_SDK/tutorials/coco2017_rgb_20x416x416x3.npy -o $VBX_SDK/tutorials/coco2017_rgb_20x416x416x3.npy -s 416 416 
fi

echo "Downloading yolo-v3-tf..."
# model details @ https://github.com/openvinotoolkit/open_model_zoo/tree/2021.4.2/models/public/yolo-v3-tf/
[ -f coco.names ] || wget -q https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names
[ -f yolov3.weights ] || wget -q http://web.archive.org/web/20210225040312/https://pjreddie.com/media/files/yolov3.weights
omz_downloader --name yolo-v3-tf
rm -rf keras-YOLOv3-model-set && git clone https://github.com/david8862/keras-YOLOv3-model-set
cd keras-YOLOv3-model-set && git checkout 56bcc2e && cd ..
python keras-YOLOv3-model-set/tools/model_converter/convert.py keras-YOLOv3-model-set/cfg/yolov3.cfg yolov3.weights yolo-v3.h5

echo "Generating TF Lite..."
tflite_quantize yolo-v3.h5 yolo-v3-tf.tflite -d $VBX_SDK/tutorials/coco2017_rgb_20x416x416x3.npy \
--scale 255. --shape 1 416 416 3

if [ -f yolo-v3-tf.tflite ]; then
   tflite_preprocess yolo-v3-tf.tflite  --scale 255
fi

if [ -f yolo-v3-tf.pre.tflite ]; then
    echo "Generating VNNX for V1000 configuration..."
    vnnx_compile -c V1000 -t yolo-v3-tf.pre.tflite -o yolo-v3-tf.vnnx
fi

if [ -f yolo-v3-tf.vnnx ]; then
    echo "Running Simulation..."
    python $VBX_SDK/example/python/yoloInfer.py yolo-v3-tf.vnnx $VBX_SDK/tutorials/test_images/dog.jpg -j yolo-v3-tf.json -v 3 -l coco.names 
    echo "C Simulation Command:"
    echo '$VBX_SDK/example/sim-c/sim-run-model yolo-v3-tf.vnnx $VBX_SDK/tutorials/test_images/dog.jpg YOLOV3'
fi

deactivate
