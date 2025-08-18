
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
if [ ! -f $VBX_SDK/tutorials/coco2017_rgb_20x513x513x3.npy ]; then
    generate_npy $VBX_SDK/tutorials/coco2017_rgb_20x416x416x3.npy -o $VBX_SDK/tutorials/coco2017_rgb_20x513x513x3.npy -s 513 513 
fi

echo "Checking for deeplabv3 files..."

# model details @ https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/deeplabv3
if [ ! -f deeplabv3.tflite ]; then
# model details @ https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/deeplabv3
omz_downloader --name deeplabv3
fi


if [ ! -f deeplabv3.tflite ]; then
   echo "Running Model Optimizer..."
   mo --input_model public/deeplabv3/deeplabv3_mnv2_pascal_train_aug/frozen_inference_graph.pb \
--input_shape=[1,513,513,3] \
--input=0:MobilenetV2/Conv/Conv2D \
--mean_values [127.5,127.5,127.5] \
--scale_values [127.5] \
--output=ArgMax \
--static_shape
fi
if [ ! -f deeplabv3.tflite ]; then
   echo "Running OpenVINO2Tensorflow..."
   openvino2tensorflow --load_dest_file_path_for_the_calib_npy $VBX_SDK/tutorials/coco2017_rgb_20x513x513x3.npy \
--keep_input_tensor_in_nchw \
--weight_replacement_config fix.json \
--model_path frozen_inference_graph.xml \
--output_full_integer_quant_tflite \
--string_formulas_for_normalization '(data - [0.,0.,0.]) / [1.,1.,1.]'
   cp saved_model/model_full_integer_quant.tflite deeplabv3.tflite
fi

if [ -f deeplabv3.tflite ]; then 
   echo "Cutting graph" 
   tflite_cut deeplabv3.tflite -c 79
   mv deeplabv3.0.tflite deeplabv3.cut.tflite 
fi

if [ -f deeplabv3.cut.tflite ]; then
   tflite_preprocess deeplabv3.cut.tflite   
fi

if [ -f deeplabv3.cut.pre.tflite ]; then
   tflite_postprocess deeplabv3.cut.pre.tflite  --post-process-layer PIXEL_VOC \
--opacity 0.8 \
--height 1080 \
--width 1920
fi

if [ -f deeplabv3.cut.pre.post.tflite ]; then
    echo "Generating VNNX for V1000 configuration..."
    vnnx_compile -c V1000 -t deeplabv3.cut.pre.post.tflite -o deeplabv3.vnnx
fi

if [ -f deeplabv3.vnnx ]; then
    echo "Running Simulation..."
    python $VBX_SDK/example/python/segmentation.py deeplabv3.vnnx $VBX_SDK/tutorials/test_images/A0PQ76.jpg --dataset VOC --inj 
    echo "C Simulation Command:"
    echo '$VBX_SDK/example/sim-c/sim-run-model deeplabv3.vnnx $VBX_SDK/tutorials/test_images/A0PQ76.jpg  '
fi

deactivate
